# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import re
from decimal import Decimal
from hashlib import sha256
from typing import Any
from urllib.parse import quote


_SAFE_LOCAL_NAME = re.compile(r"[^A-Za-z0-9._~-]+")


class RDFoxTerms:
    XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"
    XSD_DATE_TIME = "http://www.w3.org/2001/XMLSchema#dateTime"
    XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"
    XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"
    XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
    XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

    def __init__(self, base_iri: str = "https://awslabs.github.io/graphrag-toolkit/rdfox/"):
        self.base_iri = base_iri if base_iri.endswith(("/", "#")) else f"{base_iri}/"
        self.pg = f"{self.base_iri}pg/"
        self.node_ns = f"{self.base_iri}node/"
        self.edge_ns = f"{self.base_iri}edge/"
        self.prop_ns = f"{self.base_iri}prop/"
        self.type_ns = f"{self.base_iri}type/"
        self.edge_type_ns = f"{self.base_iri}edgeType/"

    def iri(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"IRI value must be a string, got {type(value).__name__}")
        return f"<{value}>"

    def local(self, value: Any) -> str:
        token = str(value)
        token = token.strip("`").strip()
        token = _SAFE_LOCAL_NAME.sub("_", token)
        return token or sha256(str(value).encode("utf-8")).hexdigest()

    def label_iri(self, label: str) -> str:
        return f"{self.type_ns}{quote(self.local(label), safe='')}"

    def predicate_iri(self, property_name: str) -> str:
        return f"{self.prop_ns}{quote(self.local(property_name), safe='')}"

    def edge_type_iri(self, relationship_type: str) -> str:
        return f"{self.edge_type_ns}{quote(self.local(relationship_type), safe='')}"

    def node_iri(self, label: str, node_id: Any) -> str:
        digest = sha256(f"{label}\0{node_id}".encode("utf-8")).hexdigest()
        return f"{self.node_ns}{digest}"

    def edge_iri(self, source_iri: str, relationship_type: str, target_iri: str, value: Any = None) -> str:
        digest = sha256(
            f"{source_iri}\0{relationship_type}\0{target_iri}\0{value}".encode("utf-8")
        ).hexdigest()
        return f"{self.edge_ns}{digest}"

    def literal(self, value: Any) -> str:
        if value is None:
            return self.typed_literal("", self.XSD_STRING)
        if isinstance(value, bool):
            return self.typed_literal("true" if value else "false", self.XSD_BOOLEAN)
        if isinstance(value, int) and not isinstance(value, bool):
            return self.typed_literal(str(value), self.XSD_INTEGER)
        if isinstance(value, float):
            return self.typed_literal(repr(value), self.XSD_DOUBLE)
        if isinstance(value, Decimal):
            return self.typed_literal(str(value), self.XSD_DECIMAL)
        if isinstance(value, datetime.datetime):
            return self.typed_literal(value.isoformat(), self.XSD_DATE_TIME)
        if isinstance(value, datetime.date):
            return self.typed_literal(value.isoformat(), self.XSD_DATE_TIME)
        if isinstance(value, (dict, list, tuple, set)):
            return self.typed_literal(json.dumps(value, sort_keys=True), self.XSD_STRING)
        return self.typed_literal(str(value), self.XSD_STRING)

    def typed_literal(self, value: str, datatype_iri: str) -> str:
        escaped = (
            value.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace('"', '\\"')
        )
        return f'"{escaped}"^^<{datatype_iri}>'

    def sparql_value_to_python(self, binding: dict[str, Any]) -> Any:
        value = binding.get("value")
        datatype = binding.get("datatype")
        binding_type = binding.get("type")

        if binding_type == "uri":
            return value
        if datatype == self.XSD_BOOLEAN:
            return value == "true"
        if datatype == self.XSD_INTEGER:
            return int(value)
        if datatype in (self.XSD_DECIMAL, self.XSD_DOUBLE):
            return float(value)
        return value
