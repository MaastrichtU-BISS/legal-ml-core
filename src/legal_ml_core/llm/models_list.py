"""Ask an OpenAI-compatible provider which chat models it offers.

``GET {base_url}/models`` with the key being checked or the owner's saved key.
The key goes only in the request headers: it is never logged, never put in an
exception message and never returned.

The transport is urllib3 on a single connection, not ``requests``: no redirect
is followed or read, nothing is decompressed, every body is read in bounded
amounts against a byte cap, and one wall-clock deadline covers connect, headers
and body together. Refusals are raised outside ``except`` blocks, so none
carries a library exception (which can hold the request headers) as context.
"""

import json
import re
import socket
import threading
from http.client import HTTPException
from time import monotonic
from urllib.parse import urlsplit

import urllib3
from urllib3.connection import HTTPConnection, HTTPSConnection

from .config import int_setting

try:  # what requests verifies against, so the key check trusts the same CAs
    import certifi

    CA_CERTS = certifi.where()
except ImportError:  # pragma: no cover - certifi comes with requests
    CA_CERTS = None

MAX_TIMEOUT_SECONDS = 15
MAX_CONNECT_SECONDS = 5
MAX_BODY_BYTES = 2 * 1024 * 1024
# Enough of a 400 to find "API key" in it, which is how Gemini rejects a key.
MAX_ERROR_BODY_BYTES = 8 * 1024
CHUNK_BYTES = 64 * 1024

OPENROUTER_HOST = "openrouter.ai"
ANTHROPIC_HOST = "api.anthropic.com"
# https://platform.claude.com/docs/en/api/models-list takes the key as a bearer
# token, as llm/client.py sends it for chat, and an ``anthropic-version`` header.
# Its list is paginated, 20 by default and 1000 at most, with ``display_name``
# beside each ``id``.
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_PAGE_LIMIT = 1000

# Ids of models that don't chat, for providers that don't say what a model
# outputs. ``tts`` only as a whole token, so ``mattshumer/...`` stays.
NON_CHAT_ID = re.compile(
    r"embed|whisper|(?:^|[^a-z0-9])tts(?:$|[^a-z0-9])|dall-e|image|audio"
    r"|moderation|realtime|transcribe",
    re.IGNORECASE,
)

MESSAGES = {
    "key_rejected": "The provider rejected this API key.",
    "provider_unreachable": "The provider could not be reached. Check the base URL and try again.",
    "listing_unsupported": "This provider doesn't list its models. Type the model id instead.",
    "saved_key_unreadable": "The stored API key can't be read. Please add it again.",
}
RATE_LIMITED = "The provider is limiting requests. Try again in a moment."

# Everything the network side can raise: urllib3's own errors (ProtocolError,
# TimeoutError and the rest), http.client's, and socket and TLS errors.
NETWORK_ERRORS = (urllib3.exceptions.HTTPError, HTTPException, OSError)


class ModelListError(Exception):
    """A refusal with a machine-readable ``code`` and a message safe to show."""

    def __init__(self, code, message=None):
        super().__init__(message or MESSAGES[code])
        self.code = code


def timeout_seconds():
    """The whole budget for one listing: at least 1 s, at most 15 s."""
    return max(1, min(int_setting("LLM_TIMEOUT_SECONDS"), MAX_TIMEOUT_SECONDS))


class _Watched:
    """A connection the watchdog can cut, whatever phase it is in.

    ``_new_conn`` keeps a duplicate of the TCP socket: TLS takes over (detaches)
    the original, but ``shutdown`` on the duplicate still ends the connection,
    which wakes a blocked connect handshake, header read or body read.
    """

    aborted = False
    _handle = None

    def _new_conn(self):
        sock = super()._new_conn()
        self._handle = sock.dup()
        if self.aborted:  # the watchdog fired while the socket was connecting
            self._cut()
        return sock

    def _cut(self):
        handle = self._handle
        if handle is not None:
            try:
                handle.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def abort(self):
        self.aborted = True
        self._cut()

    def set_read_timeout(self, seconds):
        self.timeout = seconds
        if self.sock is not None:
            self.sock.settimeout(seconds)

    def dispose(self):
        """Close the connection and the duplicate.

        Not ``close``: http.client calls that itself when a response says
        ``Connection: close``, while the response still reads the socket and the
        watchdog must still be able to cut it.
        """
        try:
            self.close()
        finally:
            handle, self._handle = self._handle, None
            if handle is not None:
                handle.close()


class _WatchedHTTPS(_Watched, HTTPSConnection):
    pass


class _WatchedHTTP(_Watched, HTTPConnection):
    """Plain http, for tests against a loopback server; the route allows https only."""


def _open(scheme, host, port, connect_timeout):
    """An unconnected connection with ``connect``, ``set_read_timeout``,
    ``request``, ``getresponse``, ``abort`` and ``dispose``. Tests replace this."""
    if scheme == "http":
        return _WatchedHTTP(host, port, timeout=connect_timeout)
    return _WatchedHTTPS(host, port, timeout=connect_timeout, ca_certs=CA_CERTS)


def _headers(host, api_key):
    # One credential header only, Bearer, for every provider.
    headers = {"Authorization": f"Bearer {api_key}"}
    if host == ANTHROPIC_HOST:
        headers["anthropic-version"] = ANTHROPIC_VERSION
    return headers


def _remaining(deadline):
    left = deadline - monotonic()
    if left <= 0:
        raise ModelListError("provider_unreachable")
    return left


def _read_capped(response, deadline, cap):
    """Up to ``cap + 1`` raw bytes of the body: a longer body is cut there."""
    body = bytearray()
    while len(body) <= cap:
        _remaining(deadline)
        chunk = response.read(min(CHUNK_BYTES, cap + 1 - len(body)), decode_content=False)
        if not chunk:
            break
        body.extend(chunk)
    return bytes(body)


def _status_refusal(status):
    if status in (401, 403):
        return ModelListError("key_rejected")
    if status == 429:
        return ModelListError("provider_unreachable", RATE_LIMITED)
    if status == 408 or status >= 500:
        return ModelListError("provider_unreachable")
    if status not in (200, 400):
        # 404, a redirect, or anything else that isn't a listing.
        return ModelListError("listing_unsupported")
    return None


def _exchange(conn, target, headers, deadline):
    """The body of a 200, or a raised ``ModelListError``. Network errors propagate."""
    conn.connect()
    conn.set_read_timeout(_remaining(deadline))
    unsendable = False
    try:
        conn.request(
            "GET",
            target,
            headers={**headers, "Accept": "application/json", "Accept-Encoding": "identity"},
            preload_content=False,
            decode_content=False,
        )
    except ValueError:
        # A header value http.client can't encode or send (UnicodeEncodeError
        # included). The route already refuses such a key when it is typed in,
        # so this can only be a saved key.
        unsendable = True
    if unsendable:
        raise ModelListError("saved_key_unreadable")

    conn.set_read_timeout(_remaining(deadline))
    response = conn.getresponse()
    try:
        refusal = _status_refusal(response.status)
        if refusal is not None:
            raise refusal
        encoding = (response.headers.get("Content-Encoding") or "").strip().lower()
        if encoding not in ("", "identity"):
            # Asked for identity; nothing is ever decompressed.
            raise ModelListError("listing_unsupported")
        if response.status == 400:
            # Gemini answers a bad key with a 400 that says "API key".
            head = _read_capped(response, deadline, MAX_ERROR_BODY_BYTES)
            if b"api key" in head.lower():
                raise ModelListError("key_rejected")
            raise ModelListError("listing_unsupported")
        declared = (response.headers.get("Content-Length") or "").strip()
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise ModelListError("listing_unsupported")
        body = _read_capped(response, deadline, MAX_BODY_BYTES)
        if len(body) > MAX_BODY_BYTES:
            raise ModelListError("listing_unsupported")
        return body
    finally:
        response.close()


def _fetch(url, headers, deadline):
    """``GET url`` within the deadline: the raw body of a 200, or ``ModelListError``.

    A watchdog cuts the connection at the deadline, so a provider that trickles
    its handshake, headers or body can't hold the request open past it.
    """
    parts = urlsplit(url)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    remaining = _remaining(deadline)
    conn = _open(parts.scheme, parts.hostname, parts.port, min(remaining, MAX_CONNECT_SECONDS))
    watchdog = threading.Timer(remaining, conn.abort)
    watchdog.daemon = True
    watchdog.start()
    body = refusal = None
    unreachable = False
    try:
        body = _exchange(conn, target, headers, deadline)
    except ModelListError as exc:
        refusal = exc  # raised without context: it came from outside any except
    except NETWORK_ERRORS:
        unreachable = True
    finally:
        watchdog.cancel()
        conn.dispose()
    # A read the watchdog cut can look like a clean end of the body.
    if unreachable or conn.aborted:
        raise ModelListError("provider_unreachable")
    if refusal is not None:
        raise refusal
    return body


def _entries(raw):
    unreadable = False
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        unreadable = True
    if unreadable:
        raise ModelListError("listing_unsupported")
    # ``{"data": [...]}``, or a bare list as Together sends.
    data = payload if isinstance(payload, list) else (
        payload.get("data") if isinstance(payload, dict) else None
    )
    if not isinstance(data, list):
        raise ModelListError("listing_unsupported")
    return [
        e
        for e in data
        if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"].strip()
    ]


def _outputs_text(entry):
    """OpenRouter's say: ``output_modalities``, else the ``modality`` string.

    Only a model that positively says it outputs text counts.
    """
    architecture = entry.get("architecture")
    if not isinstance(architecture, dict):
        return False
    outputs = architecture.get("output_modalities")
    if isinstance(outputs, list):
        return "text" in outputs
    modality = architecture.get("modality")
    if isinstance(modality, str) and "->" in modality:
        return "text" in modality.split("->", 1)[1].split("+")
    return False


def _is_chat(entry, host):
    if host == OPENROUTER_HOST:
        return _outputs_text(entry)
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
    host = urlsplit(base_url).hostname
    headers = _headers(host, api_key)
    deadline = monotonic() + timeout_seconds()

    if host == OPENROUTER_HOST:
        # OpenRouter lists its models to any key, valid or not, so ask first
        # whether it knows this one:
        # https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key
        _fetch(f"{base_url}/key", headers, deadline)

    url = f"{base_url}/models"
    if host == ANTHROPIC_HOST:
        url += f"?limit={ANTHROPIC_PAGE_LIMIT}"
    raw = _fetch(url, headers, deadline)

    models = {}
    for entry in _entries(raw):
        model_id = entry["id"]
        # A provider echoing the key back can't get it returned.
        if model_id in models or api_key in model_id or not _is_chat(entry, host):
            continue
        model = {"id": model_id}
        name = _name(entry)
        if name and name != model_id and api_key not in name:
            model["name"] = name
        models[model_id] = model
    return [models[i] for i in sorted(models)]
