"""Fail-loud error types for Iteration 9 response evaluation.

Evaluation errors are NEVER silently absorbed into scores: a
malformed replay panel, a missing judge record, or a broken
factorial relation must raise EvaluationError and halt the stage.
"""

from __future__ import annotations


class EvaluationError(RuntimeError):
    """Base class for evaluation-stage failures."""


#: Provider error codes that mean "this INPUT was rejected", as opposed to
#: "this request could not be served right now". Aliyun's MaaS gateway returns
#: ``data_inspection_failed`` with HTTP 400 and the message "Input text data
#: may contain inappropriate content" when a model's input moderation flags the
#: prompt. It is a property of the payload and of THAT model's moderation
#: policy, so it is deterministic: re-sending the identical bytes cannot
#: succeed. Measured on 2026-09-06 over the frozen 600-cell panel, judge A
#: (qwen3.8-max) refused 2 cells this way while judge B (glm-5.2) and the
#: adjudicator (kimi-k3) both returned 200 on the byte-identical payload --
#: see outputs/iteration_11/diagnostics/judge_moderation/.
INPUT_MODERATION_CODES = frozenset({
    "data_inspection_failed",
    "content_filter",
    "content_policy_violation",
})


class ProviderRejectedRequest(EvaluationError):
    """The provider refused the request itself, with a reason it states.

    Carries what ``HTTPError.__str__`` throws away. A 400 stringifies to only
    "400 Client Error: Bad Request for url: ...", so diagnosing one meant
    re-issuing the request by hand to read the body -- which is how the
    input-moderation refusal above was eventually identified, eleven retries
    and two lost 600-item judge arms later. The body is the evidence, so the
    exception keeps it.

    Attributes:
        status: HTTP status code, or None if the failure was not an HTTP
            response (a timeout or a connection drop, which stay retryable).
        code: The provider's own machine-readable error code.
        message: The provider's own human-readable error message.
        body: The raw response body, truncated.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 code: str | None = None, provider_message: str | None = None,
                 body: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.provider_message = provider_message
        self.body = body

    @property
    def is_input_moderation(self) -> bool:
        """Was the PAYLOAD refused, rather than the request being unservable?

        The distinction decides whether one cell is unjudgeable by this
        identity (record it, exclude the cell, keep going) or the whole run is
        misconfigured (an unactivated model id or an expired key refuses every
        cell identically, and recording 600 per-item refusals would bury that).
        """
        return (self.code or "") in INPUT_MODERATION_CODES

    @property
    def is_retryable(self) -> bool:
        """Could re-sending the identical request succeed?

        False for any 4xx that is not a rate limit: the request is fixed, so a
        client-side rejection is a statement about the request, not about the
        moment it was made.
        """
        if self.status is None:
            return True
        return self.status == 429 or self.status >= 500
