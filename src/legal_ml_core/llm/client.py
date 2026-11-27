"""The one LLM adapter: an OpenAI-compatible ``/chat/completions`` client.

That covers OpenRouter, llama.cpp, vLLM and most hosted providers. It sends no
tools and doesn't rely on ``response_format``, which servers support unevenly;
the output is validated by prompts.parse_suggestion instead.
"""

import math
import time
from email.utils import parsedate_to_datetime

import requests

from .config import int_setting

# A reset further away than this is a daily allowance, not a per-minute one.
DAILY_THRESHOLD_SECONDS = 15 * 60


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


def _seconds_until(value, now, parse_date):
    """Seconds from ``now`` that ``value`` names, or None when it names none."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            if not parse_date:
                return None
            try:
                when = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                return None
            if when is None:
                return None
            return when.timestamp() - now
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _reset_wait(value, now):
    """``X-RateLimit-Reset`` is an epoch time: milliseconds when over 1e12."""
    number = _seconds_until(value, now, parse_date=False)
    if number is None:
        return None
    epoch = number / 1000 if number > 1e12 else number
    return epoch - now


def _wait_from(headers, now):
    wait = _seconds_until(_header(headers, "retry-after"), now, parse_date=True)
    if wait is None:
        wait = _reset_wait(_header(headers, "x-ratelimit-reset"), now)
    return wait


def parse_rate_limit(response, now=None):
    """A ``ProviderRateLimited`` for a 429 ``response``. The body is only read, never shown."""
    now = time.time() if now is None else now
    try:
        payload = response.json()
    except Exception:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    message = error.get("message") if isinstance(error.get("message"), str) else ""
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    body_headers = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}

    wait = _wait_from(getattr(response, "headers", None), now)
    if wait is None:
        wait = _wait_from(body_headers, now)
    retry_after = max(1, math.ceil(wait)) if wait is not None else None
    lowered = message.lower()
    daily = (
        "per-day" in lowered
        or "per day" in lowered
        or (retry_after is not None and retry_after > DAILY_THRESHOLD_SECONDS)
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
            raise parse_rate_limit(response)
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
