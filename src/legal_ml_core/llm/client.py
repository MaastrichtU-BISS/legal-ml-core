"""The one LLM adapter: an OpenAI-compatible ``/chat/completions`` client.

That covers OpenRouter, llama.cpp, vLLM and most hosted providers. It sends no
tools and doesn't rely on ``response_format``, which servers support unevenly;
the output is validated by prompts.parse_suggestion instead.
"""

import math
import time
from datetime import timezone
from email.utils import parsedate_to_datetime

import requests

from .config import int_setting
from .models_list import free_without_paid_fallback

# OpenRouter allows a free model 20 requests a minute; an X-RateLimit-Limit
# above that names the daily allowance (50 or 1000) that was hit.
FREE_PER_MINUTE_REQUESTS = 20
# No provider wait is reported as longer than this.
MAX_RETRY_AFTER_SECONDS = 7 * 24 * 3600

NO_QUOTA_MESSAGE = (
    "The LLM service says this key has no quota left. "
    "Check your provider's billing, or pick another model."
)


class LLMError(Exception):
    """A provider failure, with a message that is safe to show the user."""


class ProviderRateLimited(LLMError):
    """The provider answered 429.

    ``retry_after`` is whole seconds (at least 1) or ``None`` when the provider
    didn't say. ``daily`` is True for a daily allowance, False for a short limit.
    """

    def __init__(self, retry_after=None, daily=False):
        self.retry_after = retry_after
        self.daily = daily
        super().__init__(rate_limit_message(retry_after, daily))


def _wait_words(seconds):
    if seconds > 90 * 60:
        hours = math.ceil(seconds / 3600)
        return f"{hours} hour" + ("s" if hours != 1 else "")
    if seconds > 60:
        minutes = math.ceil(seconds / 60)
        return f"{minutes} minutes"
    return "a minute"


def rate_limit_message(retry_after, daily):
    if daily:
        if retry_after is None:
            return (
                "This model's daily allowance at the provider is used up. "
                "Try again later, or pick another model."
            )
        return (
            "This model's daily allowance at the provider is used up. "
            f"Try again in {_wait_words(retry_after)}, or pick another model."
        )
    if retry_after is None:
        return (
            "The model's provider is limiting requests right now. "
            "Try again later, or pick another model."
        )
    return (
        "The model's provider is limiting requests right now. "
        f"Try again in {_wait_words(retry_after)}."
    )


def _header(headers, name):
    """Case-insensitive lookup in a mapping that may not be case-insensitive."""
    if not headers:
        return None
    try:
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == name:
                return value
    except AttributeError:
        return None
    return None


def _number(value):
    """``value`` as a finite float, or None. Any conversion failure is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    elif not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _retry_after_wait(value, now):
    """Seconds ``Retry-After`` names: a non-negative number, or an HTTP date."""
    number = _number(value)
    if number is not None:
        return number if number >= 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = parsedate_to_datetime(value.strip())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:  # ``-0000`` parses naive; HTTP dates are UTC
        when = when.replace(tzinfo=timezone.utc)
    try:
        return when.timestamp() - now
    except (OverflowError, ValueError, OSError):
        return None


def _reset_wait(value, now):
    """``X-RateLimit-Reset``: a delta in seconds below 1e9, epoch milliseconds
    from 1e12, epoch seconds in between."""
    number = _number(value)
    if number is None:
        return None
    if number < 1e9:
        return number
    if number >= 1e12:
        return number / 1000 - now
    return number - now


def _positive(wait):
    return wait if wait is not None and wait > 0 else None


def _wait_from(headers, now):
    wait = _positive(_retry_after_wait(_header(headers, "retry-after"), now))
    if wait is None:
        wait = _positive(_reset_wait(_header(headers, "x-ratelimit-reset"), now))
    return wait


def _limit_from(headers):
    number = _number(_header(headers, "x-ratelimit-limit"))
    return int(number) if number is not None and number.is_integer() else None


def _error_of(response):
    """The body's ``error`` object, or ``{}``. Nothing in it is ever shown."""
    try:
        payload = response.json()
    except Exception:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


def _metadata_of(error):
    metadata = error.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def is_insufficient_quota(response):
    """A 429 that means the key has no quota or credit left, not a rate limit.

    OpenAI says ``insufficient_quota`` in ``error.code`` or ``error.type``;
    OpenRouter relays an upstream's in ``error.metadata.error_type``.
    """
    error = _error_of(response)
    values = (error.get("code"), error.get("type"), _metadata_of(error).get("error_type"))
    return any(isinstance(v, str) and v.lower() == "insufficient_quota" for v in values)


def parse_rate_limit(response, now=None, free=False):
    """A ``ProviderRateLimited`` for a 429 ``response``. The body is only read, never shown.

    Daily when the message names OpenRouter's ``...-per-day`` limit, or, for a
    free model (``free``), when ``X-RateLimit-Limit`` is over the per-minute
    allowance. Never by how far away the reset is. An upstream provider's 429
    that OpenRouter relays (``error.metadata.provider_code``) is daily only by
    the ``per-day`` name: its limit header is that provider's, not OpenRouter's.
    """
    now = time.time() if now is None else now
    error = _error_of(response)
    message = error.get("message") if isinstance(error.get("message"), str) else ""
    metadata = _metadata_of(error)
    upstream = metadata.get("provider_code") is not None
    body_headers = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}
    response_headers = getattr(response, "headers", None)

    wait = _wait_from(response_headers, now)
    if wait is None:
        wait = _wait_from(body_headers, now)
    retry_after = (
        min(max(1, math.ceil(wait)), MAX_RETRY_AFTER_SECONDS) if wait is not None else None
    )
    limit = _limit_from(response_headers)
    if limit is None:
        limit = _limit_from(body_headers)
    daily = "per-day" in message.lower() or (
        free and not upstream and limit is not None and limit > FREE_PER_MINUTE_REQUESTS
    )
    return ProviderRateLimited(retry_after, daily)


def estimate_tokens(text):
    return max(1, len(text) // 4)


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, extra_body=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_body = extra_body or {}

    def chat(self, messages):
        """Return ``(content, total_tokens)``. Provider bodies are never surfaced."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "max_tokens": int_setting("LLM_MAX_OUTPUT_TOKENS"),
            "temperature": 0,
        }
        try:
            # Redirects stay off so a redirect can't carry the key to another host.
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=int_setting("LLM_TIMEOUT_SECONDS"),
                allow_redirects=False,
            )
        except requests.RequestException:
            raise LLMError("The LLM service could not be reached.")

        if response.status_code in (401, 403):
            raise LLMError("The LLM service rejected the API key.")
        if response.status_code == 429:
            if is_insufficient_quota(response):
                raise LLMError(NO_QUOTA_MESSAGE)
            # A paid fallback's limit header names its per-minute limit, not a
            # daily one, so only a free model with no paid extras reads it so.
            free = free_without_paid_fallback(self.base_url, self.model, self.extra_body)
            raise parse_rate_limit(response, free=free)
        if response.status_code != 200:
            raise LLMError("The LLM service returned an error.")

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("The LLM service returned an unreadable response.")
        if not isinstance(content, str):
            raise LLMError("The LLM service returned an unreadable response.")

        usage = payload.get("usage") or {}
        tokens = usage.get("total_tokens")
        if not isinstance(tokens, int):
            prompt = "".join(m["content"] for m in messages)
            tokens = estimate_tokens(prompt) + estimate_tokens(content)
        return content, tokens
