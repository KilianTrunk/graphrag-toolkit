# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.byokg_rag.graph_connectors import SPARQLKGLinker
from graphrag_toolkit.byokg_rag.graph_retrievers import GraphQueryRetriever
from graphrag_toolkit_contrib.byokg_rag.graphstore.rdfox import RDFoxSPARQLGraphStore


class _Client:
    def __init__(self):
        self.queries = []

    def query(self, sparql):
        self.queries.append(sparql)
        if "DISTINCT ?predicate" in sparql:
            return [{"predicate": "http://schema.org/knows"}]
        if "DISTINCT ?class" in sparql:
            return [{"class": "http://schema.org/Person"}]
        if "VALUES ?source" in sparql:
            return [
                {
                    "source": "https://example.com/alice",
                    "predicate": "http://schema.org/knows",
                    "object": "https://example.com/bob",
                }
            ]
        return [{"entity": "https://example.com/alice"}]


class _LLM:
    def generate(self, *args, **kwargs):
        return "<sparql>SELECT ?entity WHERE { ?entity ?p ?o }</sparql>"


def test_byokg_rdfox_store_exposes_schema_and_triplets():
    store = RDFoxSPARQLGraphStore("rdfox://localhost:12110/kg", client=_Client())

    assert store.get_schema()["graphSummary"]["edgeLabels"] == ["knows"]
    edges = store.get_one_hop_edges(["https://example.com/alice"], return_triplets=True)

    assert edges == {
        "https://example.com/alice": {
            "knows": {("https://example.com/alice", "knows", "https://example.com/bob")}
        }
    }


def test_graph_query_retriever_allows_read_only_sparql_and_blocks_updates():
    retriever = GraphQueryRetriever(graph_store=RDFoxSPARQLGraphStore("rdfox://localhost:12110/kg", client=_Client()))

    assert retriever.is_query_safe("PREFIX s: <http://schema.org/>\nSELECT ?s WHERE { ?s ?p ?o }")
    assert not retriever.is_query_safe("INSERT DATA { <s> <p> <o> }")
    assert not retriever.is_query_safe("DELETE WHERE { ?s ?p ?o }")


def test_sparql_linker_parses_sparql_tags():
    store = RDFoxSPARQLGraphStore("rdfox://localhost:12110/kg", client=_Client())
    linker = SPARQLKGLinker(graph_store=store, llm_generator=_LLM())

    parsed = linker.parse_response("<sparql>SELECT ?s WHERE { ?s ?p ?o }</sparql>")

    assert parsed["sparql"] == ["SELECT ?s WHERE { ?s ?p ?o }"]
