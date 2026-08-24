"""Source dataset adapters.

Provides the adapter classes and a name-based factory used by the
construction pipeline and CLI.
"""

from causal_mllm.adapters.base import DatasetAdapter, OnErrorMode

_ADAPTERS = ("mtmcs", "cosafe", "mtid")


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Instantiate an adapter by source dataset name.

    Args:
        name: One of 'mtmcs', 'cosafe', 'mtid'.
        **kwargs: Forwarded to the adapter constructor
            (e.g. cache_dir, media_dir, data_dir).

    Raises:
        ValueError: If the name is unknown.
    """
    key = name.lower().strip()
    if key == "mtmcs":
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        return MTMCSAdapter(**kwargs)
    if key == "cosafe":
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        return CoSafeAdapter(**kwargs)
    if key == "mtid":
        from causal_mllm.adapters.mtid import MTIDAdapter
        return MTIDAdapter(**kwargs)
    raise ValueError(
        f"Unknown source dataset '{name}'. Available: {_ADAPTERS}"
    )


__all__ = ["DatasetAdapter", "OnErrorMode", "get_adapter"]
