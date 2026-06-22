# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from hashlib import sha256
from typing import Any, Optional

from graphrag_toolkit.byokg_rag.graphstore.graphstore import GraphStore
from graphrag_toolkit_contrib.rdfox import RDFoxClient, RDFoxConnection, RDFoxTerms


class RDFoxSPARQLGraphStore(GraphStore):
    def __init__(
        self,
        graph_info: Optional[str] = None,
        *,
        endpoint_url: Optional[str] = None,
        datastore: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        bearer_token: Optional[str] = None,
        verify: bool = True,
        timeout: int = 60,
        base_iri: str = "https://awslabs.github.io/graphrag-toolkit/rdfox/byokg/",
        text_predicates: Optional[list[str]] = None,
        client: Optional[RDFoxClient] = None,
    ) -> None:
        connection = RDFoxConnection.from_connection_info(
            graph_info,
            endpoint_url=endpoint_url,
            datastore=datastore,
        )
        self.terms = RDFoxTerms(base_iri)
        self.client = client or RDFoxClient(
            connection.endpoint_url,
            connection.datastore,
            username=username,
            password=password,
            bearer_token=bearer_token,
            verify=verify,
            timeout=timeout,
            terms=self.terms,
        )
        self.text_predicates = text_predicates or [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
            "http://schema.org/name",
        ]
        self._edge_cache: dict[str, str] = {}

    def get_schema(self):
        predicates = self.client.query(
            "SELECT DISTINCT ?predicate WHERE { ?subject ?predicate ?object } ORDER BY ?predicate"
        )
        classes = self.client.query(
            "SELECT DISTINCT ?class WHERE { ?subject a ?class } ORDER BY ?class"
        )
        return {
            "graphSummary": {
                "edgeLabels": [self._display_iri(row["predicate"]) for row in predicates],
                "nodeLabels": [self._display_iri(row["class"]) for row in classes],
            }
        }

    def nodes(self):
        rows = self.client.query(
            "SELECT DISTINCT ?node WHERE { { ?node ?p ?o } UNION { ?s ?p ?node FILTER(isIRI(?node)) } } ORDER BY ?node"
        )
        return [self._display_node(row["node"]) for row in rows]

    def get_nodes(self, node_ids):
        values = self._values_clause("node", node_ids)
        rows = self.client.query(
            f"SELECT ?node ?predicate ?object WHERE {{ {values} ?node ?predicate ?object }}"
        )
        nodes: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            node = self._display_node(row["node"])
            predicate = self._display_iri(row["predicate"])
            nodes[node][predicate].append(row["object"])
        return {node: dict(properties) for node, properties in nodes.items()}

    def edges(self):
        rows = self.client.query(
            "SELECT DISTINCT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object }"
        )
        edge_ids = []
        for row in rows:
            edge_id = self._edge_id(row["subject"], row["predicate"], row["object"])
            self._edge_cache[edge_id] = row["object"]
            edge_ids.append(edge_id)
        return edge_ids

    def get_edges(self, edge_ids):
        return {
            edge_id: {"destination": self._display_node(self._edge_cache[edge_id])}
            for edge_id in edge_ids
            if edge_id in self._edge_cache
        }

    def get_one_hop_edges(self, source_node_ids, return_triplets=False):
        values = self._values_clause("source", source_node_ids)
        rows = self.client.query(
            f"SELECT ?source ?predicate ?object WHERE {{ {values} ?source ?predicate ?object }}"
        )
        expanded_edges: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for row in rows:
            source = self._display_node(row["source"])
            relation = self._display_iri(row["predicate"])
            destination = self._display_node(row["object"])
            if return_triplets:
                expanded_edges[source][relation].add((source, relation, destination))
            else:
                edge_id = self._edge_id(row["source"], row["predicate"], row["object"])
                self._edge_cache[edge_id] = row["object"]
                expanded_edges[source][relation].add(edge_id)
        return {node: dict(edges) for node, edges in expanded_edges.items()}

    def get_edge_destination_nodes(self, edge_ids):
        return {
            edge_id: [self._display_node(self._edge_cache[edge_id])]
            for edge_id in edge_ids
            if edge_id in self._edge_cache
        }

    def execute_query(self, sparql, parameters=None, read_only=False):
        if parameters:
            raise ValueError("RDFoxSPARQLGraphStore.execute_query expects a fully bound SPARQL query")
        return self.client.query(sparql)

    def get_linker_tasks(self):
        return [
            "entity-extraction",
            "path-extraction",
            "draft-answer-generation",
            "sparql",
        ]

    def _values_clause(self, variable: str, values: list[str]) -> str:
        if not values:
            return f"VALUES ?{variable} {{ }}"
        iris = []
        for value in values:
            iri = value if str(value).startswith("http://") or str(value).startswith("https://") else str(value)
            iris.append(self.terms.iri(iri))
        return f"VALUES ?{variable} {{ {' '.join(iris)} }}"

    def _display_node(self, value: Any) -> str:
        if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
            return value
        return str(value)

    def _display_iri(self, value: Any) -> str:
        value_str = str(value)
        if "#" in value_str:
            return value_str.rsplit("#", 1)[-1]
        if "/" in value_str:
            return value_str.rstrip("/").rsplit("/", 1)[-1]
        return value_str

    def _edge_id(self, source: Any, predicate: Any, destination: Any) -> str:
        return sha256(f"{source}\0{predicate}\0{destination}".encode("utf-8")).hexdigest()
