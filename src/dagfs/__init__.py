"""
Public DAGFS API.

The core implementation currently lives in `src/mlfs/` (historical naming from
the broader MLFS benchmark codebase). For external users, import from `dagfs`.
"""

from __future__ import annotations

from mlfs import DAGFSParams, dagfs

__all__ = ["DAGFSParams", "dagfs"]

