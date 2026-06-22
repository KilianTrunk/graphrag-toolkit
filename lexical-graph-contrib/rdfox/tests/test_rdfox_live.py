# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from graphrag_toolkit.byokg_rag.graph_retrievers import GTraversal
from graphrag_toolkit_contrib.byokg_rag.graphstore.rdfox import RDFoxSPARQLGraphStore


pytestmark = pytest.mark.skipif(
    not os.environ.get("RDFOX_ENDPOINT") or not os.environ.get("RDFOX_DATASTORE"),
    reason="RDFox live tests require RDFOX_ENDPOINT and RDFOX_DATASTORE",
)


def test_live_rdfox_sparql_store_round_trip():
    graph_store = RDFoxSPARQLGraphStore(
        endpoint_url=os.environ["RDFOX_ENDPOINT"],
        datastore=os.environ["RDFOX_DATASTORE"],
        username=os.environ.get("RDFOX_USERNAME"),
        password=os.environ.get("RDFOX_PASSWORD"),
        bearer_token=os.environ.get("RDFOX_BEARER_TOKEN"),
        verify=os.environ.get("RDFOX_VERIFY", "true").lower() != "false",
    )

    graph_store.client.update(
        """
        PREFIX ex: <https://example.com/graphrag-toolkit/rdfox-live/>
        INSERT DATA {
          ex:alice ex:knows ex:bob .
        }
        """
    )

    traversal = GTraversal(graph_store)
    triplets = traversal.one_hop_triplets(["https://example.com/graphrag-toolkit/rdfox-live/alice"])

    assert (
        "https://example.com/graphrag-toolkit/rdfox-live/alice",
        "knows",
        "https://example.com/graphrag-toolkit/rdfox-live/bob",
    ) in triplets
