# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from graphrag_toolkit_contrib.rdfox import RDFoxClient, RDFoxConnection, RDFoxQueryError, RDFoxTerms


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_connection_info_parses_http_and_datastore():
    connection = RDFoxConnection.from_connection_info("rdfox://localhost:12110/graphrag")

    assert connection.endpoint_url == "http://localhost:12110"
    assert connection.datastore == "graphrag"


def test_connection_info_parses_https_scheme():
    connection = RDFoxConnection.from_connection_info("rdfox+https://rdfox.example.com/prod")

    assert connection.endpoint_url == "https://rdfox.example.com"
    assert connection.datastore == "prod"


def test_query_posts_to_sparql_endpoint_and_decodes_results():
    response = _Response(
        payload={
            "head": {"vars": ["s", "count"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "https://example.com/a"},
                        "count": {
                            "type": "literal",
                            "datatype": RDFoxTerms.XSD_INTEGER,
                            "value": "3",
                        },
                    }
                ]
            },
        }
    )
    session = _Session(response)
    client = RDFoxClient("http://localhost:12110", "graphrag", session=session)

    rows = client.query("SELECT ?s WHERE { ?s ?p ?o }")

    assert rows == [{"s": "https://example.com/a", "count": 3}]
    assert session.calls[0][0] == "http://localhost:12110/datastores/graphrag/sparql"
    assert session.calls[0][1]["data"] == {"query": "SELECT ?s WHERE { ?s ?p ?o }"}


def test_update_uses_bearer_token_header():
    session = _Session(_Response(text=""))
    client = RDFoxClient("http://localhost:12110", "graphrag", bearer_token="token", session=session)

    client.update("INSERT DATA { <s> <p> <o> }")

    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer token"
    assert session.calls[0][1]["data"] == {"update": "INSERT DATA { <s> <p> <o> }"}


def test_http_errors_raise_query_error():
    client = RDFoxClient(
        "http://localhost:12110",
        "graphrag",
        session=_Session(_Response(status_code=400, text="bad query")),
    )

    with pytest.raises(RDFoxQueryError, match="HTTP 400"):
        client.query("SELECT")
