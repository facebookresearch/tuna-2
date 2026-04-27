# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base import ServerInterface
from .factory import ProviderFactory
from .protocol import Request, Response, ServerConfig
from .utils import JudgePromptBuilder, ResponseParser


def get_server(server_name: str, config: ServerConfig = None) -> ServerInterface:
    """
    Get a server instance by name.

    Args:
        server_name: Name of the server to instantiate.
        config: Optional configuration for the server.

    Returns:
        An instance of ServerInterface.
    """
    return ProviderFactory.create_provider(api_type=server_name, config=config)


__all__ = [
    "ServerInterface",
    "ServerConfig",
    "Request",
    "Response",
    "ProviderFactory",
    "JudgePromptBuilder",
    "ResponseParser",
    "get_server",
]
