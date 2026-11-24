"""Ask an OpenAI-compatible provider which chat models it offers.

``GET {base_url}/models`` with the key being checked or the owner's saved key.
The key goes only in the request headers: it is never logged, never put in an
exception message and never returned.
"""

import json
import re
from time import monotonic
from urllib.parse import urlparse

import requests

from .config import int_setting

MAX_TIMEOUT_SECONDS = 15
MAX_BODY_BYTES = 2 * 1024 * 1024
CHUNK_BYTES = 64 * 1024

OPENROUTER_HOST = "openrouter.ai"
ANTHROPIC_HOST = "api.anthropic.com"
# https://platform.claude.com/docs/en/api/models-list takes the key as a bearer
# token and an ``anthropic-version`` header. Its list is paginated, 20 by
# default and 1000 at most, with ``display_name`` beside each ``id``.
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_PAGE_LIMIT = 1000

# Ids of models that don't chat, for providers that don't say what a model outputs.
NON_CHAT_ID = re.compile(
    r"embed|whisper|tts|dall-e|image|audio|moderation|realtime|transcribe",
    re.IGNORECASE,
)

MESSAGES = {
    "key_rejected": "The provider rejected this API key.",
    "provider_unreachable": "The provider could not be reached. Check the base URL and try again.",
    "listing_unsupported": "This provider doesn't list its models. Type the model id instead.",
}


class ModelListError(Exception):
    """A refusal with a machine-readable ``code`` and a message safe to show."""

    def __init__(self, code, message=None):
        super().__init__(message or MESSAGES[code])
        self.code = code


def timeout_seconds():
    return min(int_setting("LLM_TIMEOUT_SECONDS"), MAX_TIMEOUT_SECONDS)


def _headers(host, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    if host == ANTHROPIC_HOST:
        # Bearer is documented today; x-api-key is what older keys and proxies
        # still expect. Both go to the same host, so sending both adds no exposure.
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = ANTHROPIC_VERSION
    return headers


def _read_capped(response, deadline):
    """The body, or ``listing_unsupported`` past the size cap.

    ``timeout`` in requests bounds each read, not the whole download, so a
    provider trickling bytes is cut off at the deadline as well.
    """
    body = bytearray()
    for chunk in response.iter_content(CHUNK_BYTES):
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise ModelListError("listing_unsupported")
        if monotonic() > deadline:
            raise ModelListError("provider_unreachable")
    return bytes(body)


def _fetch(url, headers, timeout):
    deadline = monotonic() + timeout
    try:
        # Redirects stay off so a redirect can't carry the key to another host.
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
    except requests.RequestException:
        raise ModelListError("provider_unreachable") from None
    try:
        status = response.status_code
        if status in (401, 403):
            raise ModelListError("key_rejected")
        if status == 429 or status >= 500:
            raise ModelListError("provider_unreachable")
        if status != 200:
            # 404, a redirect, or anything else that isn't a listing.
            raise ModelListError("listing_unsupported")
        try:
            return _read_capped(response, deadline)
        except requests.RequestException:
            raise ModelListError("provider_unreachable") from None
    finally:
        response.close()


def _entries(raw):
    try:
        payload = json.loads(raw)
    except ValueError:
        raise ModelListError("listing_unsupported") from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ModelListError("listing_unsupported")
    return [e for e in data if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"]]


def _outputs_text(entry):
    """OpenRouter's say: ``output_modalities``, else the ``modality`` string."""
    architecture = entry.get("architecture")
    if not isinstance(architecture, dict):
        return None
    outputs = architecture.get("output_modalities")
    if isinstance(outputs, list):
        return "text" in outputs
    modality = architecture.get("modality")
    if isinstance(modality, str) and "->" in modality:
        return "text" in modality.split("->", 1)[1].split("+")
    return None


def _is_chat(entry, host):
    if host == OPENROUTER_HOST:
        says = _outputs_text(entry)
        if says is not None:
            return says
    return not NON_CHAT_ID.search(entry["id"])


def _name(entry):
    for field in ("name", "display_name"):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def list_chat_models(base_url, api_key):
    """``[{id, name?}]`` sorted by id, or raise ``ModelListError``.

    The caller has already passed ``base_url`` through ``validate_byok_url``.
    """
    base_url = base_url.rstrip("/")
    host = urlparse(base_url).hostname
    url = f"{base_url}/models"
    if host == ANTHROPIC_HOST:
        url += f"?limit={ANTHROPIC_PAGE_LIMIT}"

    raw = _fetch(url, _headers(host, api_key), timeout_seconds())
    models = {}
    for entry in _entries(raw):
        if entry["id"] in models or not _is_chat(entry, host):
            continue
        model = {"id": entry["id"]}
        name = _name(entry)
        if name and name != entry["id"]:
            model["name"] = name
        models[entry["id"]] = model
    return [models[i] for i in sorted(models)]
