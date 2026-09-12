"""A node with nothing in its graph, for `assemble.py` to build on over MCP.

Importing the processors is what puts them in the node's catalog; nothing here
adds one.
"""
from streamlib import Runtime

import streamlib_doom.processors  # noqa: F401 — registers the catalog entries


def setup(rt: Runtime) -> None:
    pass
