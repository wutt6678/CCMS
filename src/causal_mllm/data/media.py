"""Media loading errors for the causal_mllm pipeline."""


class MediaLoadError(Exception):
    """Raised when a referenced media file cannot be loaded or verified.

    This is a hard failure — the pipeline must not silently convert
    a multimodal example into a text-only one.
    """

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Media load error for '{path}': {reason}")
