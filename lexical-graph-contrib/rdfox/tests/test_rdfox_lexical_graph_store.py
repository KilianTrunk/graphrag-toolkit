# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit_contrib.lexical_graph.storage.graph.rdfox import (
    RDFoxGraphStore,
    RDFoxGraphStoreFactory,
)


class _Client:
    def __init__(self):
        self.updates = []
        self.queries = []
        self.responses = []

    def update(self, sparql):
        self.updates.append(sparql)

    def query(self, sparql):
        self.queries.append(sparql)
        if self.responses:
            return self.responses.pop(0)
        return [{"boolean": True}]


def test_factory_creates_store_from_rdfox_connection_string():
    store = RDFoxGraphStoreFactory().try_create("rdfox://localhost:12110/graphrag")

    assert isinstance(store, RDFoxGraphStore)
    assert store.endpoint_url == "http://localhost:12110"
    assert store.datastore == "graphrag"


def test_unwind_merge_node_write_emits_typed_node_and_properties():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    store._client = client

    store.execute_query_with_retry(
        """// insert chunks
        UNWIND $params AS params
        MERGE (chunk:`__Chunk__`{chunkId: params.chunk_id})
        ON CREATE SET chunk.value = params.text ON MATCH SET chunk.value = params.text
        """,
        {"params": [{"chunk_id": "c1", "text": "hello"}]},
    )

    assert len(client.updates) == 1
    update = client.updates[0]
    assert "INSERT DATA" in update
    assert "type/__Chunk__" in update
    assert "prop/chunkId" in update
    assert "hello" in update


def test_unwind_merge_relationship_write_emits_edge_resource():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    store._client = client

    store.execute_query_with_retry(
        """// insert chunk-source relationships
        UNWIND $params AS params
        MERGE (chunk:`__Chunk__`{chunkId: params.chunk_id})
        MERGE (source:`__Source__`{sourceId: params.source_id})
        MERGE (chunk)-[:`__EXTRACTED_FROM__`]->(source)
        """,
        {"params": [{"chunk_id": "c1", "source_id": "s1"}]},
    )

    update = client.updates[0]
    assert "pg/Edge" in update
    assert "pg/from" in update
    assert "pg/to" in update
    assert "edgeType/__EXTRACTED_FROM__" in update


def test_delete_source_topic_lookup_uses_rdf_edge_path():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    client.responses.append([{"id": "topic-1"}])
    store._client = client

    rows = store.execute_query_with_retry(
        """// get topic ids (delete source)
        MATCH (s)<-[:`__EXTRACTED_FROM__`]-()<-[:`__MENTIONED_IN__`]-(t:`__Topic__`)
        WHERE s.sourceId = $sourceId
        RETURN DISTINCT t.topicId AS topicId LIMIT $batchSize
        """,
        {"sourceId": "source-1", "batchSize": 10},
    )

    assert rows == [{"topicId": "topic-1"}]
    assert "edgeType/__MENTIONED_IN__" in client.queries[0]
    assert "edgeType/__EXTRACTED_FROM__" in client.queries[0]


def test_delete_source_fact_lookup_uses_statement_targets():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    client.responses.append([{"id": "fact-1"}])
    store._client = client

    rows = store.execute_query_with_retry(
        """// get fact ids (delete source)
        MATCH (l)<-[:`__SUPPORTS__`]-(f)
        WHERE l.statementId IN $statementIds
        RETURN DISTINCT f.factId AS factId
        """,
        {"statementIds": ["statement-1"]},
    )

    assert rows == [{"factId": "fact-1"}]
    assert "VALUES ?targetId" in client.queries[0]
    assert "edgeType/__SUPPORTS__" in client.queries[0]
