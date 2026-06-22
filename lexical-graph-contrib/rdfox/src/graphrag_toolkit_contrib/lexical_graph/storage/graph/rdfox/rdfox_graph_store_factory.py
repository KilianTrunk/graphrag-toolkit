# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from graphrag_toolkit.lexical_graph.storage.graph import (
    GraphStore,
    GraphStoreFactoryMethod,
    get_log_formatting,
)
from graphrag_toolkit_contrib.rdfox import RDFoxConnection

from .rdfox_graph_store import RDFoxGraphStore

logger = logging.getLogger(__name__)

RDFOX_SCHEMES = ("rdfox://", "rdfox+https://")


class RDFoxGraphStoreFactory(GraphStoreFactoryMethod):
    def try_create(self, graph_info: str, **kwargs) -> GraphStore:
        endpoint_url = kwargs.pop("endpoint_url", None)
        datastore = kwargs.pop("datastore", None)

        is_rdfox_info = isinstance(graph_info, str) and graph_info.startswith(RDFOX_SCHEMES)
        if not is_rdfox_info and not (endpoint_url and datastore):
            return None

        connection = RDFoxConnection.from_connection_info(
            graph_info,
            endpoint_url=endpoint_url,
            datastore=datastore,
        )

        logger.debug(
            "Opening RDFox datastore [endpoint: %s, datastore: %s]",
            connection.endpoint_url,
            connection.datastore,
        )

        return RDFoxGraphStore(
            endpoint_url=connection.endpoint_url,
            datastore=connection.datastore,
            log_formatting=get_log_formatting(kwargs),
            **kwargs,
        )
