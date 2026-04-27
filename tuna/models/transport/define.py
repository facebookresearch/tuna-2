# pyre-unsafe
"""Compatibility shim: re-export `create_transport` from the package init.

The original tuna code split `create_transport` into a separate
`transport/define.py` module. We've folded it into `tuna.models.transport`
itself, but downstream callers (tuna.py, tuna_2r_pixel.py, tuna_2_pixel.py)
still import from this dotted path, so keep the alias.
"""

from __future__ import annotations

from tuna.models.transport import create_transport

__all__ = ["create_transport"]
