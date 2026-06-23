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


def test_unwind_merge_relationship_without_properties_emits_direct_predicate():
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
    assert "rel/extracted_from" in update
    assert "pg/Edge" not in update
    assert "pg/from" not in update
    assert "pg/to" not in update


def test_unwind_merge_nested_rows_resolve_context_values():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    store._client = client

    store.execute_query_with_retry(
        """// insert topics
        UNWIND $params AS params
        MERGE (topic:`__Topic__`{topicId: params.topic_id})
        ON CREATE SET topic.value=params.title
        WITH topic, params
        UNWIND params.chunk_ids as chunkIds
        MERGE (chunk:`__Chunk__`{chunkId: chunkIds.chunk_id})
        MERGE (topic)-[:`__MENTIONED_IN__`]->(chunk)
        """,
        {
            "params": [
                {
                    "topic_id": "topic-1",
                    "title": "Topic",
                    "chunk_ids": [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}],
                }
            ]
        },
    )

    updates = "\n".join(client.updates)
    assert "chunk-1" in updates
    assert "chunk-2" in updates
    assert "chunkIds.chunk_id" not in updates
    assert len(client.updates) == 2


def test_unwind_merge_relationship_with_properties_emits_edge_resource():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    store._client = client

    store.execute_query_with_retry(
        """// insert fact-statement relationships
        UNWIND $params AS params
        MERGE (fact:`__Fact__`{factId: params.fact_id})
        MERGE (statement:`__Statement__`{statementId: params.statement_id})
        MERGE (fact)-[:`__SUPPORTS__`{value: params.score}]->(statement)
        """,
        {"params": [{"fact_id": "f1", "statement_id": "s1", "score": 1}]},
    )

    update = client.updates[0]
    assert "pg/Edge" in update
    assert "pg/from" in update
    assert "pg/to" in update
    assert "edgeType/__SUPPORTS__" in update
    assert "prop/value" in update


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
    assert "rel/mentioned_in" in client.queries[0]
    assert "rel/extracted_from" in client.queries[0]
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
    assert "rel/supports" in client.queries[0]
    assert "edgeType/__SUPPORTS__" in client.queries[0]


def test_chunk_content_lookup_returns_values_for_node_ids():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    client.responses.append([{"content": "chunk text"}])
    store._client = client

    rows = store.execute_query_with_retry(
        """// get chunk content
        MATCH (c:`__Chunk__`)
        WHERE c.chunkId in $nodeIds
        RETURN c.value AS content
        """,
        {"nodeIds": ["chunk-1"]},
    )

    assert rows == [{"content": "chunk text"}]
    assert "VALUES ?id" in client.queries[0]
    assert "prop/chunkId" in client.queries[0]
    assert "prop/value" in client.queries[0]


def test_topic_content_lookup_uses_direct_and_reified_edges():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    client.responses.append([{"statement": "statement text", "details": ""}])
    store._client = client

    rows = store.execute_query_with_retry(
        """// get topic content
        MATCH (t:`__Topic__`)<-[:`__BELONGS_TO__`]-(s)<-[r:`__SUPPORTS__`]-()
        WHERE t.topicId = $topicId
        WITH s, count(r) AS score ORDER BY score DESC
        RETURN s.value AS statement, s.details AS details LIMIT $statementLimit
        """,
        {"topicId": "topic-1", "statementLimit": 5},
    )

    assert rows == [{"statement": "statement text", "details": ""}]
    assert "rel/belongs_to" in client.queries[0]
    assert "rel/supports" in client.queries[0]
    assert "edgeType/__BELONGS_TO__" in client.queries[0]
    assert "edgeType/__SUPPORTS__" in client.queries[0]


def test_subject_complement_lookup_matches_local_entity_by_search_string():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    client.responses.append([{"n_id": "entity-1", "c_id": "local-1"}])
    store._client = client

    rows = store.execute_query_with_retry(
        """// get complements matching subject (fact.subject)
        UNWIND $params AS params
        MATCH (n),
        (c:`__Entity__`{search_str: n.search_str, class: '__Local_Entity__'})
        WHERE n.entityId = params.nId AND n.class <> '__Local_Entity__'
        RETURN n.entityId AS n_id, c.entityId AS c_id
        """,
        {"params": [{"nId": "entity-1"}]},
    )

    assert rows == [{"n_id": "entity-1", "c_id": "local-1"}]
    assert "VALUES ?n_id" in client.queries[0]
    assert "prop/search_str" in client.queries[0]
    assert "__Local_Entity__" in client.queries[0]


def test_copy_complement_relationships_rewrites_to_real_entity():
    store = RDFoxGraphStore(endpoint_url="http://localhost:12110", datastore="graphrag")
    client = _Client()
    client.responses.append(
        [
            {
                "source": "https://awslabs.github.io/graphrag-toolkit/rdfox/node/__Entity__/source-1",
                "fact": "https://awslabs.github.io/graphrag-toolkit/rdfox/node/__Fact__/fact-1",
                "relationValue": "rel",
            }
        ]
    )
    store._client = client

    store.execute_query_with_retry(
        """// copy complement relationships to subject
        UNWIND $params AS params
        MATCH (n),
        (s)-[r:`__RELATION__`]->(c)-[:`__OBJECT__`]->(f)
        WHERE n.entityId = params.n_id AND c.entityId = params.c_id
        MERGE (s)-[:`__RELATION__`{value:r.value}]->(n)
        MERGE (n)-[:`__OBJECT__`]->(f)
        """,
        {"params": [{"n_id": "real-1", "c_id": "local-1"}]},
    )

    assert "edgeType/__RELATION__" in client.updates[0]
    assert "rel/object" in client.updates[0]
    assert "node/e3a3facac3174e42f95c072c13c1f740bbc726f94676dbada2382586f9a49b8f" in client.updates[0]
