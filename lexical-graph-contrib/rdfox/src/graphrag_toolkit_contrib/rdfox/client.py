# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote, urlparse

import requests

from .terms import RDFoxTerms


class RDFoxError(Exception):
    pass


class RDFoxQueryError(RDFoxError):
    pass


@dataclass(frozen=True)
class RDFoxConnection:
    endpoint_url: str
    datastore: str

    @staticmethod
    def from_connection_info(
        graph_info: Optional[str] = None,
        *,
        endpoint_url: Optional[str] = None,
        datastore: Optional[str] = None,
    ) -> "RDFoxConnection":
        if endpoint_url and datastore:
            return RDFoxConnection(endpoint_url=endpoint_url.rstrip("/"), datastore=datastore.strip("/"))

        if not graph_info:
            raise ValueError("RDFox connection info is required")

        parsed = urlparse(graph_info)
        if parsed.scheme not in ("rdfox", "rdfox+https"):
            raise ValueError("RDFox connection strings must start with rdfox:// or rdfox+https://")

        datastore_name = datastore or parsed.path.strip("/")
        if not datastore_name:
            raise ValueError("RDFox datastore missing from connection string")

        scheme = "https" if parsed.scheme == "rdfox+https" else "http"
        netloc = parsed.netloc
        if not netloc:
            raise ValueError("RDFox host missing from connection string")

        return RDFoxConnection(endpoint_url=f"{scheme}://{netloc}", datastore=datastore_name)


class RDFoxClient:
    def __init__(
        self,
        endpoint_url: str,
        datastore: str,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        bearer_token: Optional[str] = None,
        verify: bool = True,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
        terms: Optional[RDFoxTerms] = None,
    ) -> None:
        if username and bearer_token:
            raise ValueError("Use either username/password or bearer_token, not both")
        if username and password is None:
            raise ValueError("Password is required when username is provided")

        self.endpoint_url = endpoint_url.rstrip("/")
        self.datastore = datastore.strip("/")
        self.username = username
        self.password = password
        self.bearer_token = bearer_token
        self.verify = verify
        self.timeout = timeout
        self.session = session or requests.Session()
        self.terms = terms or RDFoxTerms()

    @property
    def sparql_url(self) -> str:
        return f"{self.endpoint_url}/datastores/{quote(self.datastore, safe='')}/sparql"

    def query(self, sparql: str) -> list[dict[str, Any]]:
        payload = self._post({"query": sparql}, accept="application/sparql-results+json")

        if "boolean" in payload:
            return [{"boolean": payload["boolean"]}]

        bindings = payload.get("results", {}).get("bindings", [])
        rows: list[dict[str, Any]] = []
        for binding in bindings:
            row = {
                key: self.terms.sparql_value_to_python(value)
                for key, value in binding.items()
            }
            rows.append(row)
        return rows

    def update(self, sparql: str) -> None:
        self._post({"update": sparql}, accept="application/json", allow_empty=True)

    def _post(self, data: dict[str, str], *, accept: str, allow_empty: bool = False) -> dict[str, Any]:
        headers = {"Accept": accept}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        auth = (self.username, self.password) if self.username else None

        response = self.session.post(
            self.sparql_url,
            data=data,
            headers=headers,
            auth=auth,
            verify=self.verify,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RDFoxQueryError(f"RDFox request failed with HTTP {response.status_code}: {response.text}")

        if allow_empty and not response.text.strip():
            return {}

        try:
            return response.json()
        except ValueError as exc:
            if allow_empty:
                return {}
            raise RDFoxQueryError(f"RDFox returned a non-JSON response: {response.text}") from exc
