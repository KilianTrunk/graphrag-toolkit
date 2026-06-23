# RDFox Support for GraphRAG Toolkit

This contributor package adds RDFox support for:

- `graphrag_toolkit.lexical_graph` as a bounded RDF/SPARQL-backed graph store.
- `graphrag_toolkit.byokg_rag` as a native SPARQL/RDF graph store.

RDFox must be provisioned separately. This package connects to an existing RDFox datastore through RDFox's SPARQL REST endpoint.

## Install

```sh
pip install -e lexical-graph-contrib/rdfox
```

## Lexical Graph

```python
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit_contrib.lexical_graph.storage.graph.rdfox import RDFoxGraphStoreFactory

GraphStoreFactory.register(RDFoxGraphStoreFactory)

graph_store = GraphStoreFactory.for_graph_store(
    "rdfox://localhost:12110/graphrag",
    username="admin",
    password="password",
    base_iri="https://example.com/graphrag/",
)
```

Connection strings use `rdfox://host:port/datastore` for HTTP and `rdfox+https://host:port/datastore` for HTTPS. You can also pass `endpoint_url` and `datastore` directly.

The lexical graph adapter supports the GraphRAG Toolkit generated Cypher subset used by ingest, versioning, deletion, and traversal-based search. It is not a general-purpose Cypher engine.

Relationships without properties are stored as direct RDF predicates, for example `rel/supports`. Relationships with properties are stored as deterministic edge resources so relationship metadata can be preserved.

## BYOKG-RAG

```python
from graphrag_toolkit.byokg_rag import ByoKGQueryEngine
from graphrag_toolkit.byokg_rag.graph_connectors import SPARQLKGLinker
from graphrag_toolkit_contrib.byokg_rag.graphstore.rdfox import RDFoxSPARQLGraphStore

graph_store = RDFoxSPARQLGraphStore(
    "rdfox://localhost:12110/my-rdf-kg",
    username="admin",
    password="password",
)

sparql_linker = SPARQLKGLinker(graph_store=graph_store, llm_generator=llm_generator)
engine = ByoKGQueryEngine(
    graph_store=graph_store,
    sparql_kg_linker=sparql_linker,
    llm_generator=llm_generator,
)
```

BYOKG-RAG uses RDFox natively through SPARQL and does not require the lexical graph RDF mapping.
