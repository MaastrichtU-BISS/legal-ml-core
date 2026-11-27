"""Ask an OpenAI-compatible provider which chat models it offers.

``GET {base_url}/models`` with the key being checked or the owner's saved key.
The key goes only in the request headers: it is never logged, never put in an
exception message and never returned.

The transport is urllib3 on a single connection, not ``requests``: no redirect
is followed or read, nothing is decompressed, every body is read in bounded
amounts against a byte cap, and one wall-clock deadline covers each connection
attempt, the TLS handshake, the headers and the body together. The DNS lookup
is not covered: it is left to the system resolver (a follow-up). Refusals are
raised outside ``except`` blocks, so none carries a library exception (which can
hold the request headers) as context. While a listing runs, the thread's records
on the ``urllib3.connection`` logger are dropped, since urllib3 logs raw header
text there when a provider's headers fail to parse.
"""

import hashlib
import json
import logging
import re
import socket
import sys
import threading
from http.client import HTTPException
from time import monotonic
from urllib.parse import urlsplit

import urllib3
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.util.connection import allowed_gai_family

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


# Set while this thread is inside a model listing.
_listing = threading.local()


class _QuietWhileListing(logging.Filter):
    """Drops this thread's urllib3.connection records during a listing only.

    urllib3 logs "Failed to parse headers" there with the raw header text, which
    can be a provider echoing the key back. Other threads, and this thread
    outside a listing, log as before.
    """

    def filter(self, record):
        return not getattr(_listing, "active", False)


def _install_log_filter():
    logger = logging.getLogger("urllib3.connection")
    if not any(isinstance(f, _QuietWhileListing) for f in logger.filters):
        logger.addFilter(_QuietWhileListing())


_install_log_filter()

DECLARED_LENGTH = re.compile(r"[0-9]{1,15}")
ALL_DIGITS = re.compile(r"[0-9]+")


def timeout_seconds():
    """The whole budget for one listing: at least 1 s, at most 15 s."""
    return max(1, min(int_setting("LLM_TIMEOUT_SECONDS"), MAX_TIMEOUT_SECONDS))


class _Watched:
    """A connection the watchdog can cut, whatever phase it is in.

    ``_new_conn`` connects within ``deadline`` itself, one address at a time,
    and keeps a duplicate of the TCP socket: TLS takes over (detaches) the
    original, but ``shutdown`` on the duplicate still ends the connection, which
    wakes a blocked TLS handshake, header read or body read. ``connect`` in
    urllib3 calls ``_new_conn`` and then wraps the socket for TLS, so
    certificate checking is urllib3's own.

    ``_lock`` keeps ``abort`` (on the watchdog thread) and ``dispose`` apart, so
    the watchdog never shuts down a descriptor number that was closed and then
    reused by another socket.
    """

    aborted = False
    deadline = None
    _handle = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def _new_conn(self):
        deadline = self.deadline
        if deadline is None:
            deadline = monotonic() + MAX_CONNECT_SECONDS
        host = self._dns_host
        if host.startswith("["):
            host = host.strip("[]")
        # Not bounded by the deadline: the system resolver can't be interrupted.
        # The families urllib3 itself asks for (IPv4 only where this host has no IPv6).
        unresolvable = False
        try:
            addresses = socket.getaddrinfo(host, self.port, allowed_gai_family(), socket.SOCK_STREAM)
        except UnicodeError:
            # A host IDNA can't encode (an empty or over-long label), say from an allowlist.
            unresolvable = True
        if unresolvable:
            raise ModelListError("provider_unreachable")
        sock = None
        error = None
        for family, kind, proto, _, address in addresses:
            if self.aborted or monotonic() >= deadline:
                raise ModelListError("provider_unreachable")
            candidate = None
            try:
                # Inside the try, as in urllib3: a family this host can't open
                # (EAFNOSUPPORT for IPv6 here) falls through to the next address.
                candidate = socket.socket(family, kind, proto)
                for option in self.socket_options or ():
                    candidate.setsockopt(*option)
                candidate.settimeout(min(MAX_CONNECT_SECONDS, max(deadline - monotonic(), 0.001)))
                if self.source_address:
                    candidate.bind(self.source_address)
                candidate.connect(address)
            except OSError as exc:
                if candidate is not None:
                    candidate.close()
                error = exc
                continue
            sock = candidate
            break
        if sock is None:
            if error is None:
                raise OSError("getaddrinfo returned no addresses")
            try:
                raise error
            finally:
                # As urllib3 does: break the exception -> traceback -> frame cycle.
                error = None
        sys.audit("http.client.connect", self, self.host, self.port)
        try:
            handle = sock.dup()
        except OSError:
            sock.close()
            raise
        with self._lock:
            self._handle = handle
            if self.aborted:  # the watchdog fired while the socket was connecting
                self._cut()
        return sock

    def _cut(self):
        """Shut the duplicate down. Call with ``_lock`` held."""
        handle = self._handle
        if handle is not None:
            try:
                handle.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def abort(self):
        with self._lock:
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
            with self._lock:
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
        if ALL_DIGITS.fullmatch(declared):
            # Leading zeros don't make a length longer: ``0000000000000000012`` is 12.
            declared = declared.lstrip("0") or "0"
        if DECLARED_LENGTH.fullmatch(declared):
            if int(declared) > MAX_BODY_BYTES:
                raise ModelListError("listing_unsupported")
        elif ALL_DIGITS.fullmatch(declared):
            # More than 15 digits: far over the cap, and too long for int().
            raise ModelListError("listing_unsupported")
        # Anything else (``²``, ``1,2``) is ignored: the capped read still holds.
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
    conn.deadline = deadline
    watchdog = threading.Timer(remaining, conn.abort)
    watchdog.daemon = True
    watchdog.start()
    body = refusal = None
    unreachable = False
    _listing.active = True
    try:
        body = _exchange(conn, target, headers, deadline)
    except ModelListError as exc:
        refusal = exc  # raised without context: it came from outside any except
    except NETWORK_ERRORS:
        unreachable = True
    finally:
        _listing.active = False
        watchdog.cancel()
        watchdog.join()
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
    entries = [
        e
        for e in data
        if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"].strip()
    ]
    if data and not entries:
        # Something came back, but not one model in it: not a model list, so not a
        # sign the key works. An empty list, or models the chat filter drops, still is.
        raise ModelListError("listing_unsupported")
    return entries


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


# OpenRouter's request allowance for ``:free`` models: 50 a day, or 1000 a day
# once the account has bought $10 of credit (the key's
# ``free_model_daily_requests`` says which; ``is_free_tier`` true means 50).
# https://openrouter.ai/docs/api-reference/limits
FREE_TIER_DAILY_REQUESTS = 50
KEY_CHECK_TIMEOUT_SECONDS = 5


def _host(base_url):
    """The hostname to compare: lowercase, without a trailing dot."""
    try:
        return (urlsplit((base_url or "").rstrip("/")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_openrouter_https(base_url):
    """``https://openrouter.ai/...`` with no user info in the URL."""
    try:
        parts = urlsplit(base_url.strip())
    except ValueError:
        return False
    return (
        parts.scheme.lower() == "https"
        and "@" not in parts.netloc
        and _host(base_url.strip()) == OPENROUTER_HOST
    )


def is_free_model(base_url, model):
    """True only for an OpenRouter model, over https, whose id ends in ``:free``.

    No other provider says which of its models cost nothing, so every other
    model counts as paid.
    """
    if not isinstance(base_url, str) or not isinstance(model, str):
        return False
    return _is_openrouter_https(base_url) and model.strip().endswith(":free")


# Request fields that let OpenRouter bill something on a ``:free`` model: a
# fallback route, plugins and web search, tools, or a preset that may add them.
PAID_EXTRAS = ("route", "plugins", "tools", "preset", "web_search_options")


def free_without_paid_fallback(base_url, model, extra_body):
    """``is_free_model``, unless ``extra_body`` lets OpenRouter bill something.

    Any of ``PAID_EXTRAS``, or ``models`` naming an id that isn't free, may be
    billed (a paid fallback, web search, a plugin), so the paid limits stay.
    """
    if not is_free_model(base_url, model):
        return False
    extra = extra_body or {}
    if not isinstance(extra, dict):
        return False
    if any(name in extra for name in PAID_EXTRAS):
        return False
    models = extra.get("models")
    if models is None:
        return True
    return isinstance(models, list) and all(is_free_model(base_url, m) for m in models)


def _count(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


# A key's allowance is asked at most once a minute per key: the estimate and
# the panel may ask again and again. Keyed by base URL and the key's SHA-256,
# never the key itself; a failed lookup isn't kept.
ALLOWANCE_CACHE_SECONDS = 60
_allowance_cache = {}
_allowance_lock = threading.Lock()
_cache_clock = monotonic  # replaced in tests


def clear_allowance_cache():
    with _allowance_lock:
        _allowance_cache.clear()


def openrouter_free_allowance(base_url, api_key):
    """``(limit, remaining)``: requests a day OpenRouter allows this key on free models.

    Asks ``GET {base_url}/key``, and only for an https OpenRouter ``base_url``.
    ``data.free_model_daily_requests`` gives both when there; otherwise
    ``data.is_free_tier`` true means ``(50, None)``. ``is_free_tier`` false
    alone says too little (an account that bought under $10 of credit still has
    50), so it is ``(None, None)``: unknown. Any failure gives ``(50, None)``,
    the lower figure. ``limit_remaining`` there is dollars of credit, not
    requests. An answer is kept for ``ALLOWANCE_CACHE_SECONDS``.
    """
    fallback = (FREE_TIER_DAILY_REQUESTS, None)
    if not isinstance(base_url, str) or not isinstance(api_key, str) or not api_key:
        return fallback
    if not _is_openrouter_https(base_url):
        return fallback
    base_url = base_url.strip().rstrip("/")
    cache_key = (base_url, hashlib.sha256(api_key.encode()).hexdigest())
    now = _cache_clock()
    with _allowance_lock:
        cached = _allowance_cache.get(cache_key)
        if cached is not None and now < cached[0]:
            return cached[1]
    answer = _ask_free_allowance(base_url, api_key)
    if answer is None:
        return fallback
    with _allowance_lock:
        for key in [k for k, (until, _) in _allowance_cache.items() if until <= now]:
            del _allowance_cache[key]
        _allowance_cache[cache_key] = (now + ALLOWANCE_CACHE_SECONDS, answer)
    return answer


def _ask_free_allowance(base_url, api_key):
    """``openrouter_free_allowance``'s lookup; None when it failed."""
    deadline = monotonic() + min(timeout_seconds(), KEY_CHECK_TIMEOUT_SECONDS)
    try:
        raw = _fetch(f"{base_url}/key", _headers(_host(base_url), api_key), deadline)
    except Exception:  # best effort: any failure is the lower figure
        return None
    try:
        data = json.loads(raw).get("data")
    except (ValueError, RecursionError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    requests = data.get("free_model_daily_requests")
    if isinstance(requests, dict):
        limit, remaining = _count(requests.get("limit")), _count(requests.get("remaining"))
        if limit is not None:
            return limit, remaining
    if data.get("is_free_tier") is False:
        return None, None
    if data.get("is_free_tier") is True:
        return FREE_TIER_DAILY_REQUESTS, None
    return None


def list_chat_models(base_url, api_key):
    """``[{id, name?, free?}]`` sorted by id, or raise ``ModelListError``.

    ``free`` is there, and true, only for a model ``is_free_model`` says is free.

    The caller has already passed ``base_url`` through ``validate_byok_url``.
    """
    base_url = base_url.rstrip("/")
    # ``openrouter.ai.`` is ``openrouter.ai``: compare without the trailing dot.
    host = (urlsplit(base_url).hostname or "").lower().rstrip(".")
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
        if is_free_model(base_url, model_id):
            model["free"] = True
        models[model_id] = model
    return [models[i] for i in sorted(models)]
