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
        if "UNWIND $PARAMS AS PARAMS" in upper and "MERGE " in upper:
            return self._execute_unwind_merge(normalized, parameters)
        if "DELETE SOURCE" in upper and "RETURN DISTINCT" in upper:
            return self._execute_delete_source_read(normalized, parameters)
        if upper.startswith("// SET VERSION INFO") or "\nSET " in upper:
            return self._execute_set_query(normalized, parameters)
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
            nodes: dict[str, dict[str, Any]] = {}
            updates: list[str] = []

            for match in node_specs:
                var = match.group("var")
                label = match.group("label")
                id_key = self._clean_property_key(match.group("key"))
                node_id = self._value(match.group("expr"), param, parameters)
                node_iri = self.terms.node_iri(label, node_id)
                nodes[var] = {"iri": node_iri, "label": label, "id_key": id_key, "id": node_id}

                updates.extend(self._node_insert_triples(node_iri, label, id_key, node_id))

            for assignment in _SET_ASSIGNMENT.finditer(cypher):
                var = assignment.group("var")
                if var not in nodes:
                    continue
                value = self._value(assignment.group("expr"), param, parameters)
                updates.append(self._property_triple(nodes[var]["iri"], assignment.group("key"), value))

            for match in rel_specs:
                src = nodes.get(match.group("src"))
                dst = nodes.get(match.group("dst"))
                if not src or not dst:
                    raise UnsupportedRDFoxQueryError(f"Relationship endpoints were not MERGEd in query: {cypher}")

                rel_type = match.group("type")
                rel_props = match.group("props")
                rel_value = self._relationship_value(rel_props, param, parameters)
                updates.extend(self._relationship_insert_triples(src["iri"], rel_type, dst["iri"], rel_value, rel_props))

            if updates:
                self.client.update(f"INSERT DATA {{\n{chr(10).join(updates)}\n}}")

        return []

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

    def _execute_read_query(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        upper = cypher.upper()
        if "delete source" in cypher and "RETURN DISTINCT" in cypher:
            return self._execute_delete_source_read(cypher, parameters)
        if "RETURN DISTINCT" in cypher and " AS " in cypher:
            return self._execute_projection_read(cypher, parameters)
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

    def _execute_structured_result_read(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if "source_id:" in cypher and "valid_from:" in cypher:
            return self._execute_source_version_read(parameters)
        if "sourceId:" in cypher and "nodeIds:" in cypher:
            return self._execute_source_node_ids_read(cypher, parameters)
        if "source:" in cypher and "topics:" in cypher:
            return self._execute_statement_grouping_read(parameters)
        raise UnsupportedRDFoxQueryError(f"Unsupported RDFox structured read query: {cypher}")

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

    def _relationship_value(self, props: Optional[str], param: dict[str, Any], parameters: dict[str, Any]) -> Any:
        if not props:
            return None
        match = re.search(r"value\s*:\s*([^}]+)", props)
        if not match:
            return None
        return self._value(match.group(1), param, parameters)

    def _value(self, expr: str, param: dict[str, Any], parameters: dict[str, Any]) -> Any:
        expr = expr.strip()
        expr = re.split(r"\s+ON\s+(?:CREATE|MATCH)\s+SET\s+", expr, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if expr.startswith("params."):
            return param.get(expr[len("params."):])
        if expr.startswith("$"):
            return parameters.get(expr[1:])
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        if expr.isdigit():
            return int(expr)
        return expr

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
