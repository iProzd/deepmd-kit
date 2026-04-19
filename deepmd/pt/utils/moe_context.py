# SPDX-License-Identifier: LGPL-3.0-or-later
"""Thread-local context for MoE EP parameters.

Provides a way to pass ProcessGroup objects to model constructors without
putting them in the config dict (which would break deepcopy).
"""

import threading
from typing import Any

_thread_local = threading.local()


def set_moe_ep_context(
    ep_group: Any | None,
    ep_rank: int,
    ep_size: int,
) -> None:
    """Set MoE EP context for the current thread.

    Parameters
    ----------
    ep_group : ProcessGroup or None
        The EP process group.
    ep_rank : int
        Rank within the EP group.
    ep_size : int
        Size of the EP group.
    """
    _thread_local.ep_group = ep_group
    _thread_local.ep_rank = ep_rank
    _thread_local.ep_size = ep_size


def get_moe_ep_context() -> tuple[Any | None, int, int]:
    """Get MoE EP context for the current thread.

    Returns
    -------
    ep_group : ProcessGroup or None
        The EP process group, or None if not set.
    ep_rank : int
        Rank within the EP group (0 if not set).
    ep_size : int
        Size of the EP group (1 if not set).
    """
    return (
        getattr(_thread_local, "ep_group", None),
        getattr(_thread_local, "ep_rank", 0),
        getattr(_thread_local, "ep_size", 1),
    )


def clear_moe_ep_context() -> None:
    """Clear MoE EP context for the current thread."""
    if hasattr(_thread_local, "ep_group"):
        del _thread_local.ep_group
    if hasattr(_thread_local, "ep_rank"):
        del _thread_local.ep_rank
    if hasattr(_thread_local, "ep_size"):
        del _thread_local.ep_size
