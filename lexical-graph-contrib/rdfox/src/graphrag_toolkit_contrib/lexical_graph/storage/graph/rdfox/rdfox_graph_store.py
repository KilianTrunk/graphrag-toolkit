# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Optional

from llama_index.core.bridge.pydantic import PrivateAttr

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore, NodeId, format_id
from graphrag_toolkit_contrib.rdfox import RDFoxClient, RDFoxTerms

logger = logging.getLogger(__name__)

_MERGE_NODE = re.compile(
    r"MERGE\s+\((?P<var>[A-Za-z_][A-Za-z0-9_]*):`(?P<label>[^`]+)`\{(?P<key>[^:}]+):\s*(?P<expr>[^}]+)\}\)",
    re.IGNORECASE,
)
_MERGE_REL = re.compile(
    r"MERGE\s+\((?P<src>[A-Za-z_][A-Za-z0-9_]*)\)-\[(?P<rel_var>[A-Za-z_][A-Za-z0-9_]*)?:?`(?P<type>[^`]+)`(?P<props>\{[^}]+\})?\]->\((?P<dst>[A-Za-z_][A-Za-z0-9_]*)\)",
    re.IGNORECASE,
)
_SET_ASSIGNMENT = re.compile(
    r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<expr>[^,\n]+)"
)
_NODE_ID_SELECTOR = re.compile(r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)")
_NESTED_UNWIND = re.compile(
    r"UNWIND\s+params\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s+as\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


class UnsupportedRDFoxQueryError(ValueError):
    pass


class RDFoxGraphStore(GraphStore):
    endpoint_url: str
    datastore: str
    username: Optional[str] = None
    password: Optional[str] = None
    bearer_token: Optional[str] = None
    verify: bool = True
    timeout: int = 60
    base_iri: str = "https://awslabs.github.io/graphrag-toolkit/rdfox/"

    _client: Optional[RDFoxClient] = PrivateAttr(default=None)
    _terms: Optional[RDFoxTerms] = PrivateAttr(default=None)

    @property
    def terms(self) -> RDFoxTerms:
        if self._terms is None:
            self._terms = RDFoxTerms(self.base_iri)
        return self._terms

    @property
    def client(self) -> RDFoxClient:
        if self._client is None:
            self._client = RDFoxClient(
                self.endpoint_url,
                self.datastore,
                username=self.username,
                password=self.password,
                bearer_token=self.bearer_token,
                verify=self.verify,
                timeout=self.timeout,
                terms=self.terms,
            )
        return self._client

    def __getstate__(self):
        self._client = None
        self._terms = None
        return super().__getstate__()

    def node_id(self, id_name: str) -> NodeId:
        return format_id(id_name)

    def property_assigment_fn(self, key: str, value: Any) -> Callable[[str], str]:
        return lambda x: x

    def init(self, graph_store=None):
        return None

    def _execute_query(self, cypher: str, parameters: Optional[dict] = None, correlation_id=None):
        parameters = parameters or {}
        query_id = uuid.uuid4().hex[:5]
        request_log_entry_parameters = self.log_formatting.format_log_entry(
            self._logging_prefix(query_id, correlation_id),
            cypher,
            parameters,
        )

        logger.debug(
            "[%s] Query: [query: %s, parameters: %s]",
            request_log_entry_parameters.query_ref,
            request_log_entry_parameters.query,
            request_log_entry_parameters.parameters,
        )

        start = time.time()
        results = self._dispatch(cypher, parameters)
        end = time.time()

        if logger.isEnabledFor(logging.DEBUG):
            response_log_entry_parameters = self.log_formatting.format_log_entry(
                self._logging_prefix(query_id, correlation_id),
                cypher,
                parameters,
                results,
            )
            logger.debug(
                "[%s] %sms Results: [%s]",
                response_log_entry_parameters.query_ref,
                int((end - start) * 1000),
                response_log_entry_parameters.results,
            )

        return results

    def _dispatch(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        normalized = self._normalize(cypher)
        upper = normalized.upper()

        if not normalized:
            return []
        if upper.startswith("CREATE ") or upper.startswith("CALL DB.INDEXES"):
            return []
        if "COPY COMPLEMENT RELATIONSHIPS" in upper:
            return self._execute_copy_complement_relationships(parameters)
        if "UNWIND $PARAMS AS PARAMS" in upper and "MERGE " in upper:
            return self._execute_unwind_merge(normalized, parameters)
        if "DELETE SOURCE" in upper and "RETURN DISTINCT" in upper:
            return self._execute_delete_source_read(normalized, parameters)
        if upper.startswith("// SET VERSION INFO") or "\nSET " in upper:
            return self._execute_set_query(normalized, parameters)
        if "DELETE COMPLEMENT RELATIONSHIPS" in upper:
            return self._execute_delete_complement_relationships(parameters)
        if "DELETE " in upper or "DETACH DELETE" in upper:
            return self._execute_delete_query(normalized, parameters)

        return self._execute_read_query(normalized, parameters)

    def _execute_unwind_merge(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        params = parameters.get("params", [])
        if not isinstance(params, list):
            raise UnsupportedRDFoxQueryError("RDFox lexical writes require parameters['params'] to be a list")

        node_specs = list(_MERGE_NODE.finditer(cypher))
        rel_specs = list(_MERGE_REL.finditer(cypher))

        if not node_specs and not rel_specs:
            raise UnsupportedRDFoxQueryError(f"Unsupported RDFox lexical write query: {cypher}")

        for param in params:
            for context in self._unwind_contexts(cypher, param):
                self._execute_unwind_merge_row(cypher, param, parameters, context, node_specs, rel_specs)

        return []

    def _execute_unwind_merge_row(
        self,
        cypher: str,
        param: dict[str, Any],
        parameters: dict[str, Any],
        context: dict[str, Any],
        node_specs: list[re.Match],
        rel_specs: list[re.Match],
    ) -> None:
        nodes: dict[str, dict[str, Any]] = {}
        updates: list[str] = []

        for match in node_specs:
            var = match.group("var")
            label = match.group("label")
            id_key = self._clean_property_key(match.group("key"))
            node_id = self._value(match.group("expr"), param, parameters, context)
            node_iri = self.terms.node_iri(label, node_id)
            nodes[var] = {"iri": node_iri, "label": label, "id_key": id_key, "id": node_id}

            updates.extend(self._node_insert_triples(node_iri, label, id_key, node_id))

        for assignment in _SET_ASSIGNMENT.finditer(cypher):
            var = assignment.group("var")
            if var not in nodes:
                continue
            value = self._value(assignment.group("expr"), param, parameters, context)
            updates.append(self._property_triple(nodes[var]["iri"], assignment.group("key"), value))

        for match in rel_specs:
            src = nodes.get(match.group("src"))
            dst = nodes.get(match.group("dst"))
            if not src or not dst:
                raise UnsupportedRDFoxQueryError(f"Relationship endpoints were not MERGEd in query: {cypher}")

            rel_type = match.group("type")
            rel_props = match.group("props")
            rel_value = self._relationship_value(rel_props, param, parameters, context)
            updates.extend(self._relationship_insert_triples(src["iri"], rel_type, dst["iri"], rel_value, rel_props))

        if updates:
            self.client.update(f"INSERT DATA {{\n{chr(10).join(updates)}\n}}")

    def _unwind_contexts(self, cypher: str, param: dict[str, Any]) -> list[dict[str, Any]]:
        nested_unwinds = list(_NESTED_UNWIND.finditer(cypher))
        if not nested_unwinds:
            return [{}]

        contexts = [{}]
        for nested_unwind in nested_unwinds:
            values = param.get(nested_unwind.group("key"), [])
            alias = nested_unwind.group("alias")
            contexts = [
                {**context, alias: value}
                for context in contexts
                for value in values
            ]
        return contexts

    def _execute_set_query(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        label, id_key, node_id = self._single_node_filter(cypher, parameters)
        node_iri = self.terms.node_iri(label, node_id)

        deletes: list[str] = []
        inserts: list[str] = []
        for assignment in _SET_ASSIGNMENT.finditer(cypher):
            predicate = self.terms.iri(self.terms.predicate_iri(assignment.group("key")))
            value = self._value(assignment.group("expr"), {}, parameters)
            deletes.append(f"{self.terms.iri(node_iri)} {predicate} ?old_{assignment.group('key')} .")
            inserts.append(self._property_triple(node_iri, assignment.group("key"), value))

        if inserts:
            self.client.update(
                "DELETE {\n"
                + "\n".join(deletes)
                + "\n}\nINSERT {\n"
                + "\n".join(inserts)
                + "\n}\nWHERE {\n"
                + f"{self.terms.iri(node_iri)} ?p ?o .\n"
                + "\n".join(f"OPTIONAL {{ {line} }}" for line in deletes)
                + "\n}"
            )
        return []

    def _execute_delete_query(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        label, id_key, node_ids = self._node_filter_values(cypher, parameters)
        if not isinstance(node_ids, list):
            node_ids = [node_ids]

        for node_id in node_ids:
            node_iri = self.terms.node_iri(label, node_id)
            self.client.update(
                "DELETE {\n"
                f"  {self.terms.iri(node_iri)} ?p ?o .\n"
                f"  ?incoming ?incomingP {self.terms.iri(node_iri)} .\n"
                "  ?edge ?edgeP ?edgeO .\n"
                "}\nWHERE {\n"
                f"  OPTIONAL {{ {self.terms.iri(node_iri)} ?p ?o . }}\n"
                f"  OPTIONAL {{ ?incoming ?incomingP {self.terms.iri(node_iri)} . }}\n"
                f"  OPTIONAL {{ ?edge <{self.terms.pg}from>|<{self.terms.pg}to> {self.terms.iri(node_iri)} . ?edge ?edgeP ?edgeO . }}\n"
                "}"
            )
        return []

    def _execute_copy_complement_relationships(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        for param in parameters.get("params", []):
            real_entity_id = param.get("n_id")
            complement_entity_id = param.get("c_id")
            if not real_entity_id or not complement_entity_id:
                continue

            real_entity = self.terms.node_iri("__Entity__", real_entity_id)
            complement = self.terms.node_iri("__Entity__", complement_entity_id)
            rows = self.client.query(
                "SELECT ?source ?fact ?relationValue WHERE {\n"
                + self._edge_match_pattern("?source", ["__RELATION__"], self.terms.iri(complement), "?relationEdge")
                + "\n"
                + self._edge_match_pattern(self.terms.iri(complement), ["__OBJECT__"], "?fact", "?objectEdge")
                + "\n"
                f"  OPTIONAL {{ ?relationEdge <{self.terms.predicate_iri('value')}> ?relationValue . }}\n"
                "}"
            )

            triples = []
            for row in rows:
                triples.extend(
                    self._relationship_insert_triples(
                        row["source"],
                        "__RELATION__",
                        real_entity,
                        row.get("relationValue"),
                        "{value: r.value}",
                    )
                )
                triples.extend(
                    self._relationship_insert_triples(
                        real_entity,
                        "__OBJECT__",
                        row["fact"],
                        None,
                        None,
                    )
                )
            if triples:
                self.client.update(f"INSERT DATA {{\n{chr(10).join(triples)}\n}}")
        return []

    def _execute_delete_complement_relationships(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        for param in parameters.get("params", []):
            complement_entity_id = param.get("c_id")
            if not complement_entity_id:
                continue
            complement = self.terms.node_iri("__Entity__", complement_entity_id)
            self.client.update(
                "DELETE {\n"
                f"  {self.terms.iri(complement)} ?p ?o .\n"
                f"  ?incoming ?incomingP {self.terms.iri(complement)} .\n"
                "  ?edge ?edgeP ?edgeO .\n"
                "}\nWHERE {\n"
                f"  OPTIONAL {{ {self.terms.iri(complement)} ?p ?o . }}\n"
                f"  OPTIONAL {{ ?incoming ?incomingP {self.terms.iri(complement)} . }}\n"
                f"  OPTIONAL {{ ?edge <{self.terms.pg}from>|<{self.terms.pg}to> {self.terms.iri(complement)} . ?edge ?edgeP ?edgeO . }}\n"
                "}"
            )
        return []

    def _execute_read_query(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        upper = cypher.upper()
        if "get complements matching subject" in cypher:
            return self._execute_matching_subject_complements_read(parameters)
        if "get subjects matching complement" in cypher:
            return self._execute_matching_complement_subjects_read(parameters)
        if "delete source" in cypher and "RETURN DISTINCT" in cypher:
            return self._execute_delete_source_read(cypher, parameters)
        if "RETURN DISTINCT" in cypher and "statementId" in cypher and " AS l" in cypher:
            return self._execute_statement_id_projection_read(cypher, parameters)
        if "RETURN DISTINCT" in cypher and " AS " in cypher:
            return self._execute_projection_read(cypher, parameters)
        if "get chunk content" in cypher and "RETURN c.value AS content" in cypher:
            return self._execute_node_property_read("__Chunk__", "chunkId", parameters.get("nodeIds", []), "value", "content")
        if "get topic content" in cypher and "RETURN s.value AS statement" in cypher:
            return self._execute_topic_content_read(parameters)
        if "RETURN {" in upper and " AS RESULT" in upper:
            return self._execute_structured_result_read(cypher, parameters)
        if "count(r) AS score" in cypher:
            return self._execute_entity_score_read(cypher, parameters)
        if "collect(distinct f.value) AS facts" in cypher:
            return self._execute_statement_facts_read(parameters)

        raise UnsupportedRDFoxQueryError(
            "RDFox lexical graph store supports GraphRAG Toolkit generated Cypher only; "
            f"unsupported query: {cypher}"
        )

    def _execute_delete_source_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        selector = self._first_return_selector(cypher)
        limit = int(parameters.get("batchSize", 0) or 0)

        if "sourceId" in parameters:
            source_iri = self.terms.node_iri("__Source__", parameters["sourceId"])
            if selector == "chunkId":
                ids = self._nodes_reaching_source("__Chunk__", "chunkId", source_iri, ["__EXTRACTED_FROM__"])
            elif selector == "topicId":
                ids = self._nodes_reaching_source("__Topic__", "topicId", source_iri, ["__MENTIONED_IN__", "__EXTRACTED_FROM__"])
            elif selector == "statementId":
                rel_path = ["__BELONGS_TO__", "__MENTIONED_IN__", "__EXTRACTED_FROM__"] if "__BELONGS_TO__" in cypher else ["__MENTIONED_IN__", "__EXTRACTED_FROM__"]
                ids = self._nodes_reaching_source("__Statement__", "statementId", source_iri, rel_path)
            else:
                raise UnsupportedRDFoxQueryError(f"Unsupported RDFox delete-source read query: {cypher}")
            ids = ids[:limit] if limit else ids
            return [{selector: node_id} for node_id in ids]

        if selector == "factId" and "statementIds" in parameters:
            ids = self._nodes_linked_to_targets(
                "__Fact__",
                "factId",
                "__Statement__",
                "statementId",
                parameters["statementIds"],
                ["__SUPPORTS__"],
            )
            return [{selector: node_id} for node_id in ids]

        if selector == "entityId" and "factIds" in parameters:
            ids = self._nodes_linked_to_targets(
                "__Entity__",
                "entityId",
                "__Fact__",
                "factId",
                parameters["factIds"],
                ["__SUBJECT__", "__OBJECT__"],
            )
            return [{selector: node_id} for node_id in ids]

        if selector == "factId" and "factIds" in parameters and "AND NOT" in cypher:
            ids = self._orphaned_nodes("__Fact__", "factId", parameters["factIds"], ["__SUPPORTS__"])
            return [{selector: node_id} for node_id in ids]

        if selector == "entityId" and "entityIds" in parameters and "AND NOT" in cypher:
            ids = self._orphaned_nodes("__Entity__", "entityId", parameters["entityIds"], ["__SUBJECT__", "__OBJECT__"])
            return [{selector: node_id} for node_id in ids]

        raise UnsupportedRDFoxQueryError(f"Unsupported RDFox delete-source read query: {cypher}")

    def _execute_matching_subject_complements_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        params = parameters.get("params", [])
        if not params:
            return []

        values = " ".join(self.terms.literal(param.get("nId")) for param in params if param.get("nId"))
        if not values:
            return []

        rows = self.client.query(
            "SELECT ?n_id ?c_id WHERE {\n"
            f"  VALUES ?n_id {{ {values} }}\n"
            f"  ?n a <{self.terms.label_iri('__Entity__')}> ;\n"
            f"    <{self.terms.predicate_iri('entityId')}> ?n_id ;\n"
            f"    <{self.terms.predicate_iri('search_str')}> ?searchStr ;\n"
            f"    <{self.terms.predicate_iri('class')}> ?nClass .\n"
            f"  ?c a <{self.terms.label_iri('__Entity__')}> ;\n"
            f"    <{self.terms.predicate_iri('entityId')}> ?c_id ;\n"
            f"    <{self.terms.predicate_iri('search_str')}> ?searchStr ;\n"
            f"    <{self.terms.predicate_iri('class')}> \"__Local_Entity__\" .\n"
            f"  FILTER(?nClass != \"__Local_Entity__\")\n"
            "}"
        )
        return rows

    def _execute_matching_complement_subjects_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        params = parameters.get("params", [])
        if not params:
            return []

        values = " ".join(
            f"({self.terms.literal(param.get('nId'))} {self.terms.literal(param.get('cId'))})"
            for param in params
            if param.get("nId") and param.get("cId")
        )
        if not values:
            return []

        rows = self.client.query(
            "SELECT ?n_id ?c_id WHERE {\n"
            f"  VALUES (?n_id ?c_id) {{ {values} }}\n"
            f"  ?n a <{self.terms.label_iri('__Entity__')}> ; <{self.terms.predicate_iri('entityId')}> ?n_id .\n"
            f"  ?c a <{self.terms.label_iri('__Entity__')}> ; <{self.terms.predicate_iri('entityId')}> ?c_id .\n"
            "}"
        )
        return rows

    def _execute_projection_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        selector = self._first_return_selector(cypher)
        label, id_key, values = self._node_filter_values(cypher, parameters)
        if not isinstance(values, list):
            values = [values]

        rows = []
        for value in values:
            node_iri = self.terms.node_iri(label, value)
            exists = self.client.query(f"ASK {{ {self.terms.iri(node_iri)} ?p ?o }}")
            if exists and exists[0].get("boolean"):
                rows.append({selector: value})
        return rows

    def _execute_statement_id_projection_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        limit = int(parameters.get("statementLimit", 100))

        if "chunk-based graph search" in cypher or "chunk-based semantic graph search" in cypher:
            return self._statement_ids_for_chunk(parameters.get("chunkId"), limit)
        if "chunk-based entity network search" in cypher:
            return self._statement_ids_for_chunk(parameters.get("nodeId"), limit)
        if "topic-based entity network search" in cypher:
            return self._statement_ids_for_topic(parameters.get("nodeId"), limit)
        if "topic-based graph search" in cypher:
            return self._statement_ids_for_topic(parameters.get("topicId"), limit)
        if "single entity-based graph search" in cypher:
            return self._statement_ids_for_entity(parameters.get("startId"), limit)
        if "multiple entity-based graph search" in cypher:
            return self._statement_ids_for_entity_pair(parameters.get("startId"), parameters.get("endIds", []), limit)

        raise UnsupportedRDFoxQueryError(f"Unsupported RDFox statement id projection query: {cypher}")

    def _statement_ids_for_chunk(self, chunk_id: Optional[str], limit: int) -> list[dict[str, Any]]:
        if not chunk_id:
            return []

        rows = self.client.query(
            "SELECT DISTINCT ?statementId WHERE {\n"
            f"  ?chunk a <{self.terms.label_iri('__Chunk__')}> ; <{self.terms.predicate_iri('chunkId')}> {self.terms.literal(chunk_id)} .\n"
            f"  ?statement a <{self.terms.label_iri('__Statement__')}> ; <{self.terms.predicate_iri('statementId')}> ?statementId .\n"
            + self._edge_match_pattern("?topic", ["__MENTIONED_IN__"], "?chunk", "?topicChunkEdge")
            + "\n"
            + self._edge_match_pattern("?statement", ["__BELONGS_TO__"], "?topic", "?statementTopicEdge")
            + "\n"
            f"}} LIMIT {limit}"
        )
        return [{"l": row["statementId"]} for row in rows]

    def _statement_ids_for_topic(self, topic_id: Optional[str], limit: int) -> list[dict[str, Any]]:
        if not topic_id:
            return []

        rows = self.client.query(
            "SELECT DISTINCT ?statementId WHERE {\n"
            f"  ?topic a <{self.terms.label_iri('__Topic__')}> ; <{self.terms.predicate_iri('topicId')}> {self.terms.literal(topic_id)} .\n"
            f"  ?statement a <{self.terms.label_iri('__Statement__')}> ; <{self.terms.predicate_iri('statementId')}> ?statementId .\n"
            + self._edge_match_pattern("?statement", ["__BELONGS_TO__"], "?topic", "?statementTopicEdge")
            + "\n"
            f"}} LIMIT {limit}"
        )
        return [{"l": row["statementId"]} for row in rows]

    def _statement_ids_for_entity(self, entity_id: Optional[str], limit: int) -> list[dict[str, Any]]:
        if not entity_id:
            return []

        rows = self.client.query(
            "SELECT DISTINCT ?statementId WHERE {\n"
            f"  ?entity a <{self.terms.label_iri('__Entity__')}> ; <{self.terms.predicate_iri('entityId')}> {self.terms.literal(entity_id)} .\n"
            f"  ?statement a <{self.terms.label_iri('__Statement__')}> ; <{self.terms.predicate_iri('statementId')}> ?statementId .\n"
            + self._edge_match_pattern("?entity", ["__SUBJECT__"], "?fact", "?subjectEdge")
            + "\n"
            + self._edge_match_pattern("?fact", ["__SUPPORTS__"], "?statement", "?supportEdge")
            + "\n"
            f"}} LIMIT {limit}"
        )
        return [{"l": row["statementId"]} for row in rows]

    def _statement_ids_for_entity_pair(self, start_id: Optional[str], end_ids: list[str], limit: int) -> list[dict[str, Any]]:
        if not start_id or not end_ids:
            return []

        values = " ".join(self.terms.literal(entity_id) for entity_id in end_ids)
        rows = self.client.query(
            "SELECT DISTINCT ?statementId WHERE {\n"
            f"  VALUES ?endEntityId {{ {values} }}\n"
            f"  ?start a <{self.terms.label_iri('__Entity__')}> ; <{self.terms.predicate_iri('entityId')}> {self.terms.literal(start_id)} .\n"
            f"  ?end a <{self.terms.label_iri('__Entity__')}> ; <{self.terms.predicate_iri('entityId')}> ?endEntityId .\n"
            f"  ?statement a <{self.terms.label_iri('__Statement__')}> ; <{self.terms.predicate_iri('statementId')}> ?statementId .\n"
            + self._edge_match_pattern("?start", ["__SUBJECT__", "__OBJECT__"], "?fact", "?startFactEdge")
            + "\n"
            + self._edge_match_pattern("?end", ["__SUBJECT__", "__OBJECT__"], "?fact", "?endFactEdge")
            + "\n"
            + self._edge_match_pattern("?fact", ["__SUPPORTS__"], "?statement", "?supportEdge")
            + "\n"
            f"}} LIMIT {limit}"
        )
        return [{"l": row["statementId"]} for row in rows]

    def _execute_node_property_read(
        self,
        label: str,
        id_key: str,
        node_ids: list[str],
        property_key: str,
        result_key: str,
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []

        values = " ".join(self.terms.literal(node_id) for node_id in node_ids)
        rows = self.client.query(
            f"SELECT ?{result_key} WHERE {{\n"
            f"  VALUES ?id {{ {values} }}\n"
            f"  ?node a <{self.terms.label_iri(label)}> ;\n"
            f"        <{self.terms.predicate_iri(id_key)}> ?id ;\n"
            f"        <{self.terms.predicate_iri(property_key)}> ?{result_key} .\n"
            "}"
        )
        return [{result_key: row[result_key]} for row in rows]

    def _execute_topic_content_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        topic_id = parameters.get("topicId")
        if not topic_id:
            return []

        limit = int(parameters.get("statementLimit", 10))
        return self.client.query(
            "SELECT ?statement (COALESCE(?rawDetails, \"\") AS ?details) WHERE {\n"
            f"  ?topic a <{self.terms.label_iri('__Topic__')}> ; <{self.terms.predicate_iri('topicId')}> {self.terms.literal(topic_id)} .\n"
            f"  ?statementNode a <{self.terms.label_iri('__Statement__')}> ; <{self.terms.predicate_iri('value')}> ?statement .\n"
            f"  OPTIONAL {{ ?statementNode <{self.terms.predicate_iri('details')}> ?rawDetails . }}\n"
            + self._edge_match_pattern("?statementNode", ["__BELONGS_TO__"], "?topic", "?belongsTo")
            + "\n"
            + self._edge_match_pattern("?factNode", ["__SUPPORTS__"], "?statementNode", "?supports")
            + "\n"
            "} GROUP BY ?statement ?rawDetails ORDER BY DESC(COUNT(?factNode)) "
            f"LIMIT {limit}"
        )

    def _execute_structured_result_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if "get entities for keywords" in cypher:
            return self._execute_entities_for_keyword_read(cypher, parameters)
        if "get entities for chunk ids" in cypher:
            return self._execute_entities_for_index_nodes_read("__Chunk__", "chunkId", parameters)
        if "get entities for topic ids" in cypher:
            return self._execute_entities_for_index_nodes_read("__Topic__", "topicId", parameters)
        if "Get statements for top chunk" in cypher:
            return self._execute_top_statement_read(cypher, parameters)
        if "Get entities for statement" in cypher:
            return self._execute_entities_for_statement_read(parameters)
        if "get next level in tree" in cypher:
            return self._execute_next_entity_level_read(parameters)
        if "expand entities: score entities by number of relations" in cypher:
            return self._execute_entities_by_id_read(parameters)
        if "source_id:" in cypher and "valid_from:" in cypher:
            return self._execute_source_version_read(parameters)
        if "sourceId:" in cypher and "nodeIds:" in cypher:
            return self._execute_source_node_ids_read(cypher, parameters)
        if "source:" in cypher and "topics:" in cypher:
            return self._execute_statement_grouping_read(parameters)
        raise UnsupportedRDFoxQueryError(f"Unsupported RDFox structured read query: {cypher}")

    def _execute_entities_for_keyword_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        keyword = parameters.get("keyword")
        if not keyword:
            return []

        filters = [
            f"?entity a <{self.terms.label_iri('__Entity__')}> .",
            f"?entity <{self.terms.predicate_iri('entityId')}> ?entityId .",
            f"?entity <{self.terms.predicate_iri('value')}> ?value .",
            f"?entity <{self.terms.predicate_iri('class')}> ?class .",
        ]
        if "STARTS WITH $keyword" in cypher:
            filters.append(f"?entity <{self.terms.predicate_iri('search_str')}> ?searchStr .")
            filters.append(f"FILTER(STRSTARTS(STR(?searchStr), STR({self.terms.literal(keyword)})))")
        else:
            filters.append(f"?entity <{self.terms.predicate_iri('search_str')}> {self.terms.literal(keyword)} .")

        classification = parameters.get("classification")
        if classification:
            if "class STARTS WITH $classification" in cypher:
                filters.append(f"FILTER(STRSTARTS(STR(?class), STR({self.terms.literal(classification)})))")
            else:
                filters.append(f"FILTER(?class = {self.terms.literal(classification)})")
        else:
            filters.append('FILTER(?class != "__Local_Entity__")')

        return self._execute_scored_entity_read("\n  ".join(filters), int(parameters.get("limit", 100)))

    def _execute_entities_for_index_nodes_read(
        self,
        index_label: str,
        index_id_key: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        node_ids = parameters.get("nodeIds", [])
        if not node_ids:
            return []

        values = " ".join(self.terms.literal(node_id) for node_id in node_ids)
        statement_relation = "__BELONGS_TO__" if index_label == "__Topic__" else "__MENTIONED_IN__"
        filters = [
            f"VALUES ?nodeId {{ {values} }}",
            f"?indexNode a <{self.terms.label_iri(index_label)}> ; <{self.terms.predicate_iri(index_id_key)}> ?nodeId .",
            f"?entity a <{self.terms.label_iri('__Entity__')}> .",
            f"?entity <{self.terms.predicate_iri('entityId')}> ?entityId .",
            f"?entity <{self.terms.predicate_iri('value')}> ?value .",
            f"?entity <{self.terms.predicate_iri('class')}> ?class .",
            'FILTER(?class != "__Local_Entity__")',
            self._edge_match_pattern("?statement", [statement_relation], "?indexNode", "?indexEdge"),
            self._edge_match_pattern("?fact", ["__SUPPORTS__"], "?statement", "?supportEdge"),
            self._edge_match_pattern("?entity", ["__SUBJECT__", "__OBJECT__"], "?fact", "?roleEdge"),
        ]
        return self._execute_scored_entity_read("\n  ".join(filters), int(parameters.get("limit", 100)))

    def _execute_top_statement_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        node_ids = parameters.get("nodeIds", [])
        if not node_ids:
            return []

        if "t.topicId" in cypher:
            index_label = "__Topic__"
            index_id_key = "topicId"
            statement_relation = "__BELONGS_TO__"
        else:
            index_label = "__Chunk__"
            index_id_key = "chunkId"
            statement_relation = "__MENTIONED_IN__"

        values = " ".join(self.terms.literal(node_id) for node_id in node_ids)
        rows = self.client.query(
            "SELECT DISTINCT ?statement ?statementId WHERE {\n"
            f"  VALUES ?nodeId {{ {values} }}\n"
            f"  ?indexNode a <{self.terms.label_iri(index_label)}> ; <{self.terms.predicate_iri(index_id_key)}> ?nodeId .\n"
            f"  ?statementNode a <{self.terms.label_iri('__Statement__')}> ;\n"
            f"    <{self.terms.predicate_iri('statementId')}> ?statementId ;\n"
            f"    <{self.terms.predicate_iri('value')}> ?statement .\n"
            + self._edge_match_pattern("?statementNode", [statement_relation], "?indexNode", "?statementEdge")
            + "\n}"
        )
        return [
            {"result": {"statement": row["statement"], "statementId": row["statementId"]}}
            for row in rows
        ]

    def _execute_entities_for_statement_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        statement_ids = parameters.get("statementIds", [])
        if not statement_ids:
            return []

        values = " ".join(self.terms.literal(statement_id) for statement_id in statement_ids)
        filters = [
            f"VALUES ?statementId {{ {values} }}",
            f"?statement a <{self.terms.label_iri('__Statement__')}> ; <{self.terms.predicate_iri('statementId')}> ?statementId .",
            f"?entity a <{self.terms.label_iri('__Entity__')}> .",
            f"?entity <{self.terms.predicate_iri('entityId')}> ?entityId .",
            f"?entity <{self.terms.predicate_iri('value')}> ?value .",
            f"?entity <{self.terms.predicate_iri('class')}> ?class .",
            'FILTER(?class != "__Local_Entity__")',
            self._edge_match_pattern("?fact", ["__SUPPORTS__"], "?statement", "?supportEdge"),
            self._edge_match_pattern("?entity", ["__SUBJECT__", "__OBJECT__"], "?fact", "?roleEdge"),
        ]
        return self._execute_scored_entity_read("\n  ".join(filters), int(parameters.get("limit", 100)))

    def _execute_entities_by_id_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        entity_ids = parameters.get("entityIds", [])
        if not entity_ids:
            return []

        values = " ".join(self.terms.literal(entity_id) for entity_id in entity_ids)
        filters = [
            f"VALUES ?entityId {{ {values} }}",
            f"?entity a <{self.terms.label_iri('__Entity__')}> .",
            f"?entity <{self.terms.predicate_iri('entityId')}> ?entityId .",
            f"?entity <{self.terms.predicate_iri('value')}> ?value .",
            f"?entity <{self.terms.predicate_iri('class')}> ?class .",
        ]
        return self._execute_scored_entity_read("\n  ".join(filters), int(parameters.get("limit", 100)))

    def _execute_next_entity_level_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        entity_ids = parameters.get("entityIds", [])
        if not entity_ids:
            return []

        excluded_entity_ids = parameters.get("excludeEntityIds", [])
        values = " ".join(self.terms.literal(entity_id) for entity_id in entity_ids)
        excluded_filter = ""
        if excluded_entity_ids:
            excluded = ", ".join(self.terms.literal(entity_id) for entity_id in excluded_entity_ids)
            excluded_filter = f"FILTER(?otherEntityId NOT IN ({excluded}))"

        rows = self.client.query(
            "SELECT ?entityId ?value ?class ?otherEntityId (COUNT(?scoreEdge) AS ?score) WHERE {\n"
            f"  VALUES ?entityId {{ {values} }}\n"
            f"  ?entity a <{self.terms.label_iri('__Entity__')}> ;\n"
            f"    <{self.terms.predicate_iri('entityId')}> ?entityId ;\n"
            f"    <{self.terms.predicate_iri('value')}> ?value ;\n"
            f"    <{self.terms.predicate_iri('class')}> ?class .\n"
            f"  ?other a <{self.terms.label_iri('__Entity__')}> ;\n"
            f"    <{self.terms.predicate_iri('entityId')}> ?otherEntityId ;\n"
            f"    <{self.terms.predicate_iri('class')}> ?otherClass .\n"
            f"  FILTER(?otherClass != \"__Local_Entity__\")\n"
            f"  {excluded_filter}\n"
            + self._edge_match_pattern("?entity", ["__RELATION__"], "?other", "?relationEdge")
            + "\n"
            + self._edge_match_pattern("?other", ["__SUBJECT__", "__OBJECT__"], "?target", "?scoreEdge")
            + "\n"
            "} GROUP BY ?entityId ?value ?class ?otherEntityId ORDER BY DESC(?score)"
        )

        neighbours_by_entity: dict[str, dict[str, Any]] = {}
        limit = int(parameters.get("numNeighbours", 5))
        for row in rows:
            result = neighbours_by_entity.setdefault(
                row["entityId"],
                {
                    "entity": {"entityId": row["entityId"], "value": row["value"], "class": row["class"]},
                    "others": [],
                },
            )
            if len(result["others"]) < limit:
                result["others"].append(row["otherEntityId"])

        return [{"result": result} for result in neighbours_by_entity.values()]

    def _execute_scored_entity_read(self, filters: str, limit: int) -> list[dict[str, Any]]:
        rows = self.client.query(
            "SELECT ?entityId ?value ?class (COUNT(?scoreEdge) AS ?score) WHERE {\n"
            f"  {filters}\n"
            + self._edge_match_pattern("?entity", ["__SUBJECT__", "__OBJECT__"], "?scoreTarget", "?scoreEdge")
            + "\n"
            "} GROUP BY ?entityId ?value ?class ORDER BY DESC(?score) "
            f"LIMIT {limit}"
        )
        return [
            {
                "result": {
                    "entity": {"entityId": row["entityId"], "value": row["value"], "class": row["class"]},
                    "score": row["score"],
                }
            }
            for row in rows
        ]

    def _execute_entity_score_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        keyword = parameters.get("keyword")
        classification = parameters.get("classification")
        limit = parameters.get("limit", 10)
        filters = [f"?entity <{self.terms.predicate_iri('search_str')}> {self.terms.literal(keyword)} ."]
        if classification:
            filters.append(f"?entity <{self.terms.predicate_iri('class')}> {self.terms.literal(classification)} .")

        rows = self.client.query(
            "SELECT ?entityId (COUNT(?edge) AS ?score) WHERE {\n"
            f"  ?entity a <{self.terms.label_iri('__Entity__')}> .\n"
            f"  {' '.join(filters)}\n"
            f"  ?entity <{self.terms.predicate_iri('entityId')}> ?entityId .\n"
            + self._edge_match_pattern("?entity", ["__SUBJECT__", "__OBJECT__"], "?target", "?edge")
            + "\n"
            "} GROUP BY ?entityId ORDER BY DESC(?score) "
            f"LIMIT {int(limit)}"
        )
        return rows

    def _execute_statement_facts_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        statement_ids = parameters.get("statementIds", [])
        rows: list[dict[str, Any]] = []
        for statement_id in statement_ids:
            statement_iri = self.terms.node_iri("__Statement__", statement_id)
            facts = self.client.query(
                "SELECT DISTINCT ?fact WHERE {\n"
                + self._edge_match_pattern("?factNode", ["__SUPPORTS__"], self.terms.iri(statement_iri), "?edge")
                + "\n"
                f"  ?factNode <{self.terms.predicate_iri('value')}> ?fact .\n"
                "}"
            )
            rows.append({"statementId": statement_id, "facts": [row["fact"] for row in facts]})
        return rows

    def _execute_source_version_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self.client.query(
            "SELECT ?source_id ?valid_from ?valid_to WHERE {\n"
            f"  ?source a <{self.terms.label_iri('__Source__')}> ; <{self.terms.predicate_iri('sourceId')}> ?source_id .\n"
            f"  OPTIONAL {{ ?source <{self.terms.predicate_iri('valid_from')}> ?valid_from . }}\n"
            f"  OPTIONAL {{ ?source <{self.terms.predicate_iri('valid_to')}> ?valid_to . }}\n"
            "}"
        )
        return [
            {"result": {"source_id": row["source_id"], "valid_from": row.get("valid_from"), "valid_to": row.get("valid_to")}}
            for row in rows
        ]

    def _execute_source_node_ids_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        source_ids = parameters.get("sourceIds", [])
        index_label = "__Chunk__"
        id_key = "chunkId"
        rel_path = ["__EXTRACTED_FROM__"]
        if "topicId" in cypher:
            index_label = "__Topic__"
            id_key = "topicId"
            rel_path = ["__MENTIONED_IN__", "__EXTRACTED_FROM__"]
        if "statementId" in cypher:
            index_label = "__Statement__"
            id_key = "statementId"
            rel_path = ["__BELONGS_TO__", "__MENTIONED_IN__", "__EXTRACTED_FROM__"]

        output = []
        for source_id in source_ids:
            source = self.terms.node_iri("__Source__", source_id)
            rows = self._nodes_reaching_source(index_label, id_key, source, rel_path)
            output.append({"result": {"sourceId": source_id, "nodeIds": rows}})
        return output

    def _execute_statement_grouping_read(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        statement_ids = parameters.get("statementIds", [])
        results_by_source: dict[str, dict[str, Any]] = {}

        for statement_id in statement_ids:
            row = self._statement_context(statement_id)
            if not row:
                continue
            source_id = row["source"]["sourceId"]
            source_result = results_by_source.setdefault(
                source_id,
                {"score": 0, "source": row["source"], "topics": {}},
            )
            topic = source_result["topics"].setdefault(
                row["topicId"],
                {"topic": row["topic"], "topicId": row["topicId"], "chunks": [], "statements": []},
            )
            if row["chunk"] not in topic["chunks"]:
                topic["chunks"].append(row["chunk"])
            topic["statements"].append(row["statement"])
            source_result["score"] += 1

        results = []
        for source_result in results_by_source.values():
            source_result["topics"] = list(source_result["topics"].values())
            results.append({"result": source_result})
        return sorted(results, key=lambda item: item["result"]["score"], reverse=True)

    def _statement_context(self, statement_id: str) -> Optional[dict[str, Any]]:
        statement = self.terms.node_iri("__Statement__", statement_id)
        rows = self.client.query(
            "SELECT ?statement ?details ?topic ?topicId ?chunkId ?sourceId ?sourceProp ?sourceValue WHERE {\n"
            f"  {self.terms.iri(statement)} <{self.terms.predicate_iri('statementId')}> ?statementId ; <{self.terms.predicate_iri('value')}> ?statement .\n"
            f"  OPTIONAL {{ {self.terms.iri(statement)} <{self.terms.predicate_iri('details')}> ?details . }}\n"
            + self._edge_match_pattern(self.terms.iri(statement), ["__BELONGS_TO__"], "?topicNode", "?e1")
            + "\n"
            f"  ?topicNode <{self.terms.predicate_iri('topicId')}> ?topicId ; <{self.terms.predicate_iri('value')}> ?topic .\n"
            + self._edge_match_pattern(self.terms.iri(statement), ["__MENTIONED_IN__"], "?chunkNode", "?e2")
            + "\n"
            f"  ?chunkNode <{self.terms.predicate_iri('chunkId')}> ?chunkId .\n"
            + self._edge_match_pattern("?chunkNode", ["__EXTRACTED_FROM__"], "?sourceNode", "?e3")
            + "\n"
            f"  ?sourceNode <{self.terms.predicate_iri('sourceId')}> ?sourceId .\n"
            f"  OPTIONAL {{ ?sourceNode ?sourceProp ?sourceValue . FILTER(STRSTARTS(STR(?sourceProp), \"{self.terms.prop_ns}\")) }}\n"
            "}"
        )
        if not rows:
            return None

        source = {"sourceId": rows[0]["sourceId"], "metadata": {}, "versioning": {}}
        for row in rows:
            prop = row.get("sourceProp")
            if prop:
                source["metadata"][prop.rsplit("/", 1)[-1]] = row.get("sourceValue")

        return {
            "source": source,
            "topic": rows[0].get("topic"),
            "topicId": rows[0].get("topicId"),
            "chunk": {"chunkId": rows[0].get("chunkId"), "value": None, "metadata": {}},
            "statement": {
                "statementId": statement_id,
                "statement": rows[0].get("statement"),
                "facts": [],
                "details": rows[0].get("details"),
                "chunkId": rows[0].get("chunkId"),
                "score": 0,
            },
        }

    def _nodes_reaching_source(self, label: str, id_key: str, source_iri: str, rel_path: list[str]) -> list[str]:
        nodes = ["?node"] + [f"?mid{i}" for i in range(1, len(rel_path))] + [self.terms.iri(source_iri)]
        edge_patterns = [
            self._edge_match_pattern(nodes[i], [rel_type], nodes[i + 1], f"?edge{i}")
            for i, rel_type in enumerate(rel_path)
        ]
        sparql = (
            "SELECT DISTINCT ?id WHERE {\n"
            f"  ?node a <{self.terms.label_iri(label)}> ; <{self.terms.predicate_iri(id_key)}> ?id .\n"
            + "\n".join(edge_patterns)
            + "\n}"
        )
        return [row["id"] for row in self.client.query(sparql)]

    def _nodes_linked_to_targets(
        self,
        label: str,
        id_key: str,
        target_label: str,
        target_id_key: str,
        target_ids: list[str],
        rel_types: list[str],
    ) -> list[str]:
        if not target_ids:
            return []

        values = " ".join(self.terms.literal(target_id) for target_id in target_ids)
        rows = self.client.query(
            "SELECT DISTINCT ?id WHERE {\n"
            f"  VALUES ?targetId {{ {values} }}\n"
            f"  ?target a <{self.terms.label_iri(target_label)}> ; <{self.terms.predicate_iri(target_id_key)}> ?targetId .\n"
            f"  ?node a <{self.terms.label_iri(label)}> ; <{self.terms.predicate_iri(id_key)}> ?id .\n"
            + self._edge_match_pattern("?node", rel_types, "?target", "?edge")
            + "\n"
            "}"
        )
        return [row["id"] for row in rows]

    def _orphaned_nodes(self, label: str, id_key: str, node_ids: list[str], rel_types: list[str]) -> list[str]:
        orphaned = []
        for node_id in node_ids:
            node_iri = self.terms.node_iri(label, node_id)
            links = self.client.query(
                "ASK {\n"
                + self._edge_match_pattern(self.terms.iri(node_iri), rel_types, "?target", "?edge")
                + "\n"
                "}"
            )
            if not links or not links[0].get("boolean"):
                orphaned.append(node_id)
        return orphaned

    def _node_insert_triples(self, node_iri: str, label: str, id_key: str, node_id: Any) -> list[str]:
        return [
            f"{self.terms.iri(node_iri)} a {self.terms.iri(self.terms.label_iri(label))} .",
            self._property_triple(node_iri, id_key, node_id),
        ]

    def _relationship_insert_triples(
        self,
        source_iri: str,
        rel_type: str,
        target_iri: str,
        value: Any,
        rel_props: Optional[str],
    ) -> list[str]:
        if rel_props is None:
            predicate = self.terms.iri(self.terms.relationship_iri(rel_type))
            return [
                f"{self.terms.iri(source_iri)} {predicate} {self.terms.iri(target_iri)} ."
            ]

        edge_iri = self.terms.edge_iri(source_iri, rel_type, target_iri, value)
        triples = [
            f"{self.terms.iri(edge_iri)} a <{self.terms.pg}Edge> .",
            f"{self.terms.iri(edge_iri)} <{self.terms.pg}from> {self.terms.iri(source_iri)} .",
            f"{self.terms.iri(edge_iri)} <{self.terms.pg}to> {self.terms.iri(target_iri)} .",
            f"{self.terms.iri(edge_iri)} <{self.terms.pg}edgeType> {self.terms.iri(self.terms.edge_type_iri(rel_type))} .",
        ]
        if value is not None:
            triples.append(self._property_triple(edge_iri, "value", value))
        return triples

    def _edge_match_pattern(self, source: str, rel_types: list[str], target: str, edge_var: str) -> str:
        predicate_var = f"?{edge_var.lstrip('?')}Predicate"
        direct_predicates = ", ".join(f"<{self.terms.relationship_iri(rel_type)}>" for rel_type in rel_types)
        edge_types = ", ".join(f"<{self.terms.edge_type_iri(rel_type)}>" for rel_type in rel_types)
        return (
            f"  {{ {source} {predicate_var} {target} .\n"
            f"    FILTER({predicate_var} IN ({direct_predicates}))\n"
            f"    BIND({predicate_var} AS {edge_var}) }}\n"
            "  UNION\n"
            f"  {{ {edge_var} <{self.terms.pg}from> {source} ; "
            f"<{self.terms.pg}edgeType> ?{edge_var.lstrip('?')}Type ; "
            f"<{self.terms.pg}to> {target} .\n"
            f"    FILTER(?{edge_var.lstrip('?')}Type IN ({edge_types})) }}"
        )

    def _property_triple(self, subject_iri: str, key: str, value: Any) -> str:
        return f"{self.terms.iri(subject_iri)} {self.terms.iri(self.terms.predicate_iri(key))} {self.terms.literal(value)} ."

    def _relationship_value(
        self,
        props: Optional[str],
        param: dict[str, Any],
        parameters: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        if not props:
            return None
        match = re.search(r"value\s*:\s*([^}]+)", props)
        if not match:
            return None
        return self._value(match.group(1), param, parameters, context)

    def _value(
        self,
        expr: str,
        param: dict[str, Any],
        parameters: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        expr = expr.strip()
        expr = re.split(r"\s+ON\s+(?:CREATE|MATCH)\s+SET\s+", expr, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if expr.startswith("params."):
            return self._nested_value(param, expr[len("params."):])
        context = context or {}
        context_match = re.match(r"(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.(?P<path>.+)", expr)
        if context_match and context_match.group("alias") in context:
            return self._nested_value(context[context_match.group("alias")], context_match.group("path"))
        if expr.startswith("$"):
            return parameters.get(expr[1:])
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        if expr.isdigit():
            return int(expr)
        return expr

    def _nested_value(self, data: Any, path: str) -> Any:
        value = data
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def _single_node_filter(self, cypher: str, parameters: dict[str, Any]) -> tuple[str, str, Any]:
        label, id_key, values = self._node_filter_values(cypher, parameters)
        if isinstance(values, list):
            if len(values) != 1:
                raise UnsupportedRDFoxQueryError("Expected one node id for RDFox SET query")
            values = values[0]
        return label, id_key, values

    def _node_filter_values(self, cypher: str, parameters: dict[str, Any]) -> tuple[str, str, Any]:
        match = re.search(r"MATCH\s+\((?P<var>\w+)(?::`(?P<label>[^`]+)`)?\)", cypher, re.IGNORECASE)
        if not match:
            raise UnsupportedRDFoxQueryError(f"Cannot find node MATCH clause in query: {cypher}")

        var = match.group("var")
        label = match.group("label") or self._label_from_id_key(cypher)

        where = re.search(
            rf"{re.escape(var)}\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<op>=|IN|in)\s+\$(?P<param>[A-Za-z_][A-Za-z0-9_]*)",
            cypher,
        )
        if not where:
            where = re.search(
                rf"{re.escape(var)}\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*params\.(?P<param>[A-Za-z_][A-Za-z0-9_]*)",
                cypher,
            )
        if not where:
            raise UnsupportedRDFoxQueryError(f"Cannot find supported node id filter in query: {cypher}")

        return label, where.group("key"), parameters.get(where.group("param"))

    def _label_from_id_key(self, cypher: str) -> str:
        for label, key in (
            ("__Source__", "sourceId"),
            ("__Chunk__", "chunkId"),
            ("__Topic__", "topicId"),
            ("__Statement__", "statementId"),
            ("__Fact__", "factId"),
            ("__Entity__", "entityId"),
        ):
            if key in cypher:
                return label
        raise UnsupportedRDFoxQueryError(f"Cannot infer label from query: {cypher}")

    def _first_return_selector(self, cypher: str) -> str:
        match = re.search(r"AS\s+([A-Za-z_][A-Za-z0-9_]*)", cypher)
        if not match:
            raise UnsupportedRDFoxQueryError(f"Cannot find return selector in query: {cypher}")
        return match.group(1)

    def _clean_property_key(self, key: str) -> str:
        key = key.strip()
        key = key.strip("`")
        if "." in key:
            return key.split(".")[-1]
        return key

    def _normalize(self, cypher: str) -> str:
        return "\n".join(line.rstrip() for line in cypher.strip().splitlines() if line.strip())
