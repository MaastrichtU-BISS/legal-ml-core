"""Free models (OpenRouter ``:free``) and the provider's rate limits (BAIGLE #225)."""

import json
from email.utils import format_datetime
from datetime import datetime, timezone

import pytest

from legal_ml_core import LLMCredential, LLMSuggestion, db, free_model_for, is_free_model
from legal_ml_core.llm import client as llm_client, suggest
from legal_ml_core.llm.client import ProviderRateLimited, parse_rate_limit
from legal_ml_core.llm.suggest import process_llm_annotation_task
from legal_ml_core.queue_contract import llm_job_status_key
from tests.conftest import TEST_USER_ID
from tests.test_llm_routes import (  # noqa: F401 - fixtures used by name
    DESTINATION,
    SECRET,
    FakeProvider,
    _ack,
    _dataset,
    _list_models,
    _save_key,
    _start_job,
    _suggest,
    gated,
    listing,
    llm_app,
    outsider,
    provider,
)

OPENROUTER = "https://openrouter.ai/api/v1"
FREE = "meta-llama/llama-3.3-70b-instruct:free"
NOW = 1_800_000_000.0


class Limited:
    """A 429 as ``parse_rate_limit`` reads it."""

    def __init__(self, headers=None, body=None):
        self.status_code = 429
        self.headers = headers or {}
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _openrouter_body(message="Rate limit exceeded: free-models-per-minute", headers=None):
    return {"error": {"code": 429, "message": message, "metadata": {"headers": headers or {}}}}


# Parsing the provider's 429


def test_retry_after_in_seconds():
    exc = parse_rate_limit(Limited({"Retry-After": "12"}), now=NOW)
    assert (exc.retry_after, exc.daily) == (12, False)
    assert str(exc) == "The model's provider is limiting requests right now. Try again in a minute."


def test_retry_after_is_rounded_up_and_at_least_one():
    assert parse_rate_limit(Limited({"retry-after": "2.1"}), now=NOW).retry_after == 3
    assert parse_rate_limit(Limited({"Retry-After": "0.2"}), now=NOW).retry_after == 1
    # A wait of nothing names no wait: the caller's default applies.
    assert parse_rate_limit(Limited({"Retry-After": "0"}), now=NOW).retry_after is None


def test_retry_after_as_an_http_date():
    when = datetime.fromtimestamp(NOW + 300, tz=timezone.utc)
    exc = parse_rate_limit(Limited({"Retry-After": format_datetime(when, usegmt=True)}), now=NOW)
    assert (exc.retry_after, exc.daily) == (300, False)
    assert "Try again in 5 minutes." in str(exc)


@pytest.mark.parametrize("reset", [str(int((NOW + 45) * 1000)), str(int(NOW + 45))])
def test_x_ratelimit_reset_in_milliseconds_or_seconds(reset):
    exc = parse_rate_limit(Limited({"X-RateLimit-Reset": reset}), now=NOW)
    assert (exc.retry_after, exc.daily) == (45, False)


def test_openrouter_headers_inside_the_body():
    body = _openrouter_body(headers={"X-RateLimit-Reset": str(int((NOW + 30) * 1000))})
    exc = parse_rate_limit(Limited(body=body), now=NOW)
    assert (exc.retry_after, exc.daily) == (30, False)


def test_response_headers_come_before_the_body():
    body = _openrouter_body(headers={"Retry-After": "600"})
    assert parse_rate_limit(Limited({"Retry-After": "7"}, body), now=NOW).retry_after == 7


def test_a_per_day_message_is_daily():
    body = _openrouter_body(
        "Rate limit exceeded: free-models-per-day",
        {"X-RateLimit-Reset": str(int((NOW + 5 * 3600) * 1000))},
    )
    exc = parse_rate_limit(Limited(body=body), now=NOW)
    assert exc.daily is True
    assert exc.retry_after == 5 * 3600
    assert str(exc) == (
        "This model's daily allowance at the provider is used up. "
        "Try again in 5 hours, or pick another model."
    )


def test_a_far_reset_alone_is_not_daily():
    exc = parse_rate_limit(Limited({"Retry-After": "3600"}, _openrouter_body()), now=NOW)
    assert (exc.retry_after, exc.daily) == (3600, False)


def test_the_prose_per_day_is_not_the_limit_name():
    body = _openrouter_body("You may send 50 requests per day")
    assert parse_rate_limit(Limited(body=body), now=NOW).daily is False


@pytest.mark.parametrize(
    "limit, free, daily",
    [("50", True, True), ("1000", True, True), ("20", True, False), ("1000", False, False)],
)
def test_a_free_models_limit_header_names_the_allowance_hit(limit, free, daily):
    # OpenRouter's real body: the message doesn't say which limit.
    body = _openrouter_body("Rate limit exceeded", {"X-RateLimit-Limit": limit})
    assert parse_rate_limit(Limited(body=body), now=NOW, free=free).daily is daily
    in_headers = Limited({"X-RateLimit-Limit": limit}, _openrouter_body("Rate limit exceeded"))
    assert parse_rate_limit(in_headers, now=NOW, free=free).daily is daily


def test_the_client_says_whether_its_model_is_free(monkeypatch):
    answer = Limited({"X-RateLimit-Limit": "50"}, _openrouter_body("Rate limit exceeded"))
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: answer)
    for model, daily in ((FREE, True), ("paid/model", False)):
        with pytest.raises(ProviderRateLimited) as caught:
            llm_client.OpenAICompatibleClient(OPENROUTER, "k", model).chat(
                [{"role": "user", "content": "hi"}]
            )
        assert caught.value.daily is daily


def test_a_huge_number_in_the_body_is_absent():
    body = _openrouter_body(headers={"Retry-After": 10**400, "X-RateLimit-Limit": 10**400})
    exc = parse_rate_limit(Limited(body=body), now=NOW, free=True)
    assert (exc.retry_after, exc.daily) == (None, False)


def test_a_very_long_wait_is_capped_at_a_week():
    exc = parse_rate_limit(Limited({"Retry-After": "1e50"}), now=NOW)
    assert exc.retry_after == 7 * 24 * 3600


def test_a_negative_retry_after_is_absent_and_the_reset_is_used():
    exc = parse_rate_limit(Limited({"Retry-After": "-5", "X-RateLimit-Reset": "60"}), now=NOW)
    assert exc.retry_after == 60  # a small reset is a delta in seconds


def test_a_past_reset_falls_back_to_the_body():
    body = _openrouter_body(headers={"Retry-After": "30"})
    exc = parse_rate_limit(Limited({"X-RateLimit-Reset": str(int(NOW - 10))}, body), now=NOW)
    assert exc.retry_after == 30
    assert parse_rate_limit(Limited({"X-RateLimit-Reset": str(int(NOW - 10))}), now=NOW).retry_after is None


def test_an_http_date_without_a_zone_is_utc():
    when = datetime.fromtimestamp(NOW + 300, tz=timezone.utc)
    text = when.strftime("%a, %d %b %Y %H:%M:%S -0000")
    assert parse_rate_limit(Limited({"Retry-After": text}), now=NOW).retry_after == 300


def test_an_unknown_reset():
    exc = parse_rate_limit(Limited(body={"error": "slow down"}), now=NOW)
    assert (exc.retry_after, exc.daily) == (None, False)
    assert "Try again later, or pick another model." in str(exc)
    daily = parse_rate_limit(Limited(body=_openrouter_body("free-models-per-day")), now=NOW)
    assert (daily.retry_after, daily.daily) == (None, True)
    assert str(daily) == (
        "This model's daily allowance at the provider is used up. "
        "Try again later, or pick another model."
    )


@pytest.mark.parametrize("value", ["soon", "", "nan", "inf"])
def test_unreadable_headers_are_unknown(value):
    exc = parse_rate_limit(Limited({"Retry-After": value, "X-RateLimit-Reset": value}), now=NOW)
    assert exc.retry_after is None


def test_the_provider_body_is_never_in_the_message():
    body = _openrouter_body("secret provider detail per minute")
    assert "secret" not in str(parse_rate_limit(Limited(body=body), now=NOW))


def test_the_client_raises_the_typed_error(monkeypatch):
    monkeypatch.setattr(
        llm_client.requests, "post", lambda *a, **k: Limited({"Retry-After": "9"})
    )
    with pytest.raises(ProviderRateLimited) as caught:
        llm_client.OpenAICompatibleClient("https://x.test", "k", "m").chat(
            [{"role": "user", "content": "hi"}]
        )
    assert caught.value.retry_after == 9


# Free detection


@pytest.mark.parametrize(
    "base_url, model, free",
    [
        (OPENROUTER, FREE, True),
        ("https://OpenRouter.AI./api/v1/", FREE, True),
        (OPENROUTER, "meta-llama/llama-3.3-70b-instruct", False),
        (OPENROUTER, "free/model", False),
        ("https://api.openai.com/v1", "gpt:free", False),
        ("https://openrouter.ai.evil.test/v1", FREE, False),
        ("http://openrouter.ai/api/v1", FREE, False),
        ("https://user:pw@openrouter.ai/api/v1", FREE, False),
        (None, FREE, False),
        (OPENROUTER, None, False),
    ],
)
def test_is_free_model(base_url, model, free):
    assert is_free_model(base_url, model) is free


def test_free_model_for_an_own_key_never_decrypts(app):
    with app.app_context():
        db.session.add(
            LLMCredential(
                owner_id=TEST_USER_ID, encrypted_key="garbage", base_url=OPENROUTER, model=FREE
            )
        )
        db.session.commit()
        assert free_model_for(TEST_USER_ID, {}) is True
        LLMCredential.query.filter_by(owner_id=TEST_USER_ID).update({"model": "paid/model"})
        db.session.commit()
        assert free_model_for(TEST_USER_ID, {}) is False


def test_free_model_for_the_shared_key(app):
    claims = {"email": "a@um.test", "email_verified": True}
    with app.app_context():
        assert free_model_for(TEST_USER_ID, claims) is False
        app.config["LLM_SHARED_ENDPOINT"] = {
            **app.config["LLM_SHARED_ENDPOINT"], "base_url": OPENROUTER, "model": FREE
        }
        assert free_model_for(TEST_USER_ID, claims) is True
        # Not allowed the shared key: nothing would be used, so nothing is free.
        assert free_model_for(TEST_USER_ID, {"email": "a@other.test", "email_verified": True}) is False


def test_free_model_for_never_raises(app, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("db down")

    with app.app_context():
        monkeypatch.setattr(type(LLMCredential.query), "filter_by", broken)
        assert free_model_for(TEST_USER_ID, {}) is False


def test_free_model_for_rolls_back_a_failed_query(app, monkeypatch):
    rolled = []
    with app.app_context():
        monkeypatch.setattr(
            type(LLMCredential.query), "filter_by", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
        )
        original = db.session.rollback
        monkeypatch.setattr(db.session, "rollback", lambda: (rolled.append(1), original()))
        assert free_model_for(TEST_USER_ID, {}) is False
    assert rolled == [1]


@pytest.mark.parametrize(
    "extra_body, free",
    [
        ({"provider": {"zdr": True}}, True),
        ({"models": [FREE, "other/model:free"]}, True),
        ({"models": [FREE, "paid/model"]}, False),
        ({"route": "fallback"}, False),
        ({"models": "paid/model"}, False),
        ({"plugins": [{"id": "web"}]}, False),
        ({"tools": []}, False),
        ({"preset": "@preset/x"}, False),
        ({"web_search_options": {}}, False),
    ],
)
def test_a_shared_model_with_a_paid_fallback_is_not_free(client, app, gated, extra_body, free):
    app.config["LLM_SHARED_ENDPOINT"] = {
        **app.config["LLM_SHARED_ENDPOINT"], "base_url": OPENROUTER, "model": FREE,
        "extra_body": extra_body,
    }
    assert client.get("/api/llm/status", headers=gated).get_json()["free_model"] is free
    with app.app_context():
        assert free_model_for(TEST_USER_ID, {"email": "a@um.test", "email_verified": True}) is free


def test_listing_marks_free_models(client, outsider, listing):
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": true}}')
    listing.payload = {
        "data": [
            {"id": FREE, "name": "Llama free", "architecture": {"output_modalities": ["text"]}},
            {"id": "a/paid", "architecture": {"output_modalities": ["text"]}},
        ]
    }
    resp = _list_models(client, outsider, {"base_url": OPENROUTER, "api_key": SECRET})
    assert resp.get_json()["models"] == [
        {"id": "a/paid"},
        {"id": FREE, "name": "Llama free", "free": True},
    ]


def test_listing_elsewhere_never_marks_free(client, outsider, listing):
    listing.payload = {"data": [{"id": "x:free"}]}
    assert _list_models(client, outsider).get_json()["models"] == [{"id": "x:free"}]


# The route's 429


class LimitedProvider(FakeProvider):
    """``requests.post`` answering from a list of ``(status, headers, body)``, then 200s."""

    def __init__(self, answers=()):
        super().__init__()
        self.answers = list(answers)
        self.headers = {}
        self._limited_body = None

    def json(self):
        if self.status_code == 429:
            if self._limited_body is None:
                raise ValueError("no body")
            return self._limited_body
        return super().json()

    def __call__(self, url, json=None, headers=None, **kwargs):
        super().__call__(url, json=json, headers=headers, **kwargs)
        status, self.headers, self._limited_body = (
            self.answers.pop(0) if self.answers else (200, {}, None)
        )
        self.status_code = status
        return self


@pytest.fixture
def limited(monkeypatch):
    fake = LimitedProvider()
    monkeypatch.setattr(llm_client.requests, "post", fake)
    return fake


PER_MINUTE = (429, {"Retry-After": "15"}, _openrouter_body())
PER_DAY = (429, {"Retry-After": "7200"}, _openrouter_body("free-models-per-day"))


def test_a_provider_429_is_a_429_with_retry_after(client, app, gated, limited):
    limited.answers = [PER_MINUTE]
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    resp = _suggest(client, dataset_id, task_ids, gated)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "15"
    assert resp.get_json() == {
        "error": "The model's provider is limiting requests right now. Try again in a minute.",
        "code": "provider_rate_limited",
        "retry_after": 15,
        "daily": False,
    }
    assert len(limited.calls) == 1  # the API never waits


def test_a_daily_429_without_a_reset_has_no_retry_after_header(client, app, gated, limited):
    limited.answers = [(429, {}, _openrouter_body("free-models-per-day"))]
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    resp = _suggest(client, dataset_id, task_ids, gated)
    assert resp.status_code == 429
    assert "Retry-After" not in resp.headers
    body = resp.get_json()
    assert (body["code"], body["retry_after"], body["daily"]) == ("provider_rate_limited", None, True)


def test_other_provider_errors_keep_their_mapping(client, app, gated, limited):
    limited.answers = [(401, {}, None)]
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    resp = _suggest(client, dataset_id, task_ids, gated)
    assert resp.status_code == 502
    assert resp.get_json() == {"error": "The LLM service rejected the API key.", "code": "llm_failed"}


# A free shared model


@pytest.fixture
def free_shared(app):
    app.config["LLM_SHARED_ENDPOINT"] = {
        **app.config["LLM_SHARED_ENDPOINT"], "base_url": OPENROUTER, "model": FREE
    }


def test_a_free_shared_model_ignores_a_spent_cap_but_is_still_metered(
    client, app, gated, provider, free_shared
):
    calls = []

    class Spent:
        def remaining(self, owner_id):
            return 0

        def record(self, owner_id, tokens, key):
            calls.append(tokens)

    app.config["USAGE_METER"] = Spent()
    dataset_id, task_ids = _dataset(app, tasks=2)
    _ack(client, dataset_id, gated)
    resp = _suggest(client, dataset_id, task_ids, gated)
    assert resp.status_code == 200
    assert len(resp.get_json()["suggestions"]) == 2
    assert calls == [100, 100]


def test_a_free_shared_model_with_the_cap_at_zero(client, app, gated, provider, free_shared):
    app.config["LLM_MONTHLY_TOKEN_CAP"] = 0
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    assert _suggest(client, dataset_id, task_ids, gated).status_code == 200


def test_a_free_shared_model_survives_a_meter_that_cannot_record(
    client, app, gated, provider, free_shared
):
    class Unrecordable:
        def remaining(self, owner_id):
            raise ConnectionError("down")

        def record(self, owner_id, tokens, key):
            raise ConnectionError("down")

    app.config["USAGE_METER"] = Unrecordable()
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    assert _suggest(client, dataset_id, task_ids, gated).status_code == 200


def test_a_failed_record_is_rolled_back_so_the_suggestion_is_saved(
    client, app, gated, provider, free_shared
):
    from legal_ml_core import AnnotationTask

    class Poisoning:
        """Flushes a row that breaks a constraint, leaving the session needing a rollback."""

        def remaining(self, owner_id):
            return None

        def record(self, owner_id, tokens, key):
            db.session.add(AnnotationTask(dataset_id=None, text=None))
            db.session.flush()

    app.config["USAGE_METER"] = Poisoning()
    dataset_id, task_ids = _dataset(app, tasks=2)
    _ack(client, dataset_id, gated)
    resp = _suggest(client, dataset_id, task_ids, gated)
    assert resp.status_code == 200, resp.get_json()
    assert len(resp.get_json()["suggestions"]) == 2
    with app.app_context():
        assert LLMSuggestion.query.count() == 2


def test_a_paid_shared_model_still_stops_at_the_cap(client, app, gated, provider):
    app.config["LLM_MONTHLY_TOKEN_CAP"] = 0
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    assert _suggest(client, dataset_id, task_ids, gated).status_code == 429
    assert provider.calls == []


def test_status_says_whether_the_model_is_free(client, app, gated):
    assert client.get("/api/llm/status", headers=gated).get_json()["free_model"] is False
    app.config["LLM_SHARED_ENDPOINT"] = {
        **app.config["LLM_SHARED_ENDPOINT"], "base_url": OPENROUTER, "model": FREE
    }
    assert client.get("/api/llm/status", headers=gated).get_json()["free_model"] is True


# The estimate


def _estimate(client, dataset_id, headers):
    resp = client.post(f"/api/llm/datasets/{dataset_id}/annotate/estimate", json={}, headers=headers)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


@pytest.mark.parametrize(
    "key_answer, daily_limit, daily_remaining, over",
    [
        ((200, b'{"data": {"is_free_tier": true, "limit_remaining": 3}}'), 50, None, False),
        # A paid account may have bought under $10: the allowance isn't known.
        ((200, b'{"data": {"is_free_tier": false}}'), None, None, False),
        (
            (200, b'{"data": {"is_free_tier": true, "free_model_daily_requests": '
                  b'{"used": 998, "limit": 1000, "remaining": 2}}}'),
            1000,
            2,
            True,
        ),
        (
            (200, b'{"data": {"is_free_tier": true, "free_model_daily_requests": '
                  b'{"used": 0, "limit": 1000, "remaining": 1000}}}'),
            1000,
            1000,
            False,
        ),
        (
            (200, b'{"data": {"is_free_tier": false, "free_model_daily_requests": '
                  b'{"limit": -1, "remaining": "x"}}}'),
            None,
            None,
            False,
        ),
        ((500, b"oops"), 50, None, False),
        ((200, b"not json"), 50, None, False),
    ],
)
def test_a_free_model_estimate_names_the_daily_allowance(
    client, app, gated, listing, free_shared, key_answer, daily_limit, daily_remaining, over
):
    app.config["LLM_MONTHLY_TOKEN_CAP"] = 0
    listing.responses["/key"] = key_answer
    dataset_id, _ = _dataset(app, tasks=3)
    body = _estimate(client, dataset_id, gated)
    assert body["free_model"] is True
    assert body["within_budget"] is True
    assert body["provider_daily_limit"] == daily_limit
    assert body["provider_daily_remaining"] == daily_remaining
    assert body["provider_allowance_shared"] is True
    assert body["over_provider_limit"] is over
    assert [c["url"] for c in listing.calls] == [f"{OPENROUTER}/key"]
    assert listing.calls[0]["headers"]["Authorization"] == "Bearer shared-key"


def test_a_failing_key_lookup_is_the_lower_figure(client, app, gated, listing, free_shared):
    listing.error = OSError("unreachable")
    dataset_id, _ = _dataset(app, tasks=1)
    body = _estimate(client, dataset_id, gated)
    assert (body["provider_daily_limit"], body["provider_daily_remaining"]) == (50, None)


@pytest.mark.parametrize(
    "base_url",
    ["http://openrouter.ai/api/v1", "https://llm.test/v1", "https://u:p@openrouter.ai/api/v1"],
)
def test_the_allowance_is_only_asked_of_openrouter_over_https(listing, base_url):
    from legal_ml_core.llm.models_list import openrouter_free_allowance

    assert openrouter_free_allowance(base_url, SECRET) == (50, None)
    assert listing.calls == []


def test_a_paid_model_estimate_never_asks_for_the_key(client, app, gated, listing):
    dataset_id, _ = _dataset(app, tasks=3)
    body = _estimate(client, dataset_id, gated)
    assert (body["free_model"], body["provider_daily_limit"], body["over_provider_limit"]) == (
        False,
        None,
        False,
    )
    assert body["provider_daily_remaining"] is None
    assert listing.calls == []


def test_an_own_free_key_asks_with_its_own_key(client, app, outsider, listing, monkeypatch):
    from legal_ml_core.llm import access

    monkeypatch.setattr(access, "_is_public", lambda host: True)
    resp = client.put(
        "/api/llm/key",
        json={"api_key": SECRET, "base_url": OPENROUTER, "model": FREE},
        headers=outsider,
    )
    assert resp.status_code == 200
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": true}}')
    dataset_id, _ = _dataset(app, tasks=51)
    body = _estimate(client, dataset_id, outsider)
    assert (body["free_model"], body["provider_daily_limit"], body["over_provider_limit"]) == (
        True,
        50,
        True,
    )
    # An own key's allowance is the user's own, not the platform's.
    assert body["provider_allowance_shared"] is False
    assert listing.calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_a_job_over_the_provider_limit_still_starts(client, app, gated, listing, free_shared, fake_redis):
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": true}}')
    dataset_id, _ = _dataset(app, tasks=51)
    _ack(client, dataset_id, gated)
    assert _estimate(client, dataset_id, gated)["over_provider_limit"] is True
    listing.calls.clear()
    resp = client.post(
        f"/api/llm/datasets/{dataset_id}/annotate", json={"confirm_estimate": True}, headers=gated
    )
    assert resp.status_code == 202
    estimate = resp.get_json()["estimate"]
    # The user saw the allowance in the estimate: Start doesn't ask /key again.
    assert estimate["free_model"] is True
    assert estimate["provider_daily_limit"] is None
    assert estimate["provider_daily_remaining"] is None
    assert estimate["over_provider_limit"] is False
    assert listing.calls == []
    assert fake_redis.llen("llm_queue") == 1


# The worker


@pytest.fixture
def slept(monkeypatch):
    waits = []
    monkeypatch.setattr(suggest, "_sleep", waits.append)
    return waits


def _run_job(client, app, gated, fake_redis, tasks=3):
    dataset_id, _ = _dataset(app, tasks=tasks)
    _ack(client, dataset_id, gated)
    job_id = _start_job(client, dataset_id, gated)
    process_llm_annotation_task(json.loads(fake_redis.rpop("llm_queue")), fake_redis)
    return client.get(f"/api/llm/jobs/{job_id}", headers=gated).get_json(), job_id


def test_the_worker_waits_out_a_per_minute_limit_and_carries_on(
    client, app, gated, fake_redis, limited, slept, monkeypatch
):
    seen = []
    original = suggest._set_status

    def recording(redis_client, job_id, **fields):
        if "waiting_seconds" in fields:
            seen.append(fields["waiting_seconds"])
        original(redis_client, job_id, **fields)

    monkeypatch.setattr(suggest, "_set_status", recording)
    limited.answers = [(200, {}, None), PER_MINUTE, (429, {}, None)]
    body, _ = _run_job(client, app, gated, fake_redis)
    assert body["status"] == "completed"
    assert (body["done"], body["created"]) == (3, 3)
    assert slept == [15, 20]
    assert seen == [15, None, 20, None]
    assert body["waiting_seconds"] is None


def test_a_retry_after_too_long_to_wait_stops_at_once(client, app, gated, fake_redis, limited, slept):
    limited.answers = [(429, {"Retry-After": "600"}, _openrouter_body())]
    body, _ = _run_job(client, app, gated, fake_redis, tasks=1)
    # Retrying before then is sure to fail, and spends the free daily allowance.
    assert body["status"] == "provider_limited"
    assert (body["retry_after"], body["daily"]) == (600, False)
    assert slept == []
    assert len(limited.calls) == 1
    limited.answers = [(429, {"Retry-After": "120"}, _openrouter_body())]
    body, _ = _run_job(client, app, gated, fake_redis, tasks=1)
    assert body["status"] == "completed"
    assert slept == [120]


def test_the_worker_gives_up_after_three_waits(client, app, gated, fake_redis, limited, slept):
    limited.answers = [(200, {}, None)] + [PER_MINUTE] * 4
    body, _ = _run_job(client, app, gated, fake_redis)
    assert slept == [15, 15, 15]
    assert body["status"] == "provider_limited"
    assert (body["retry_after"], body["daily"]) == (15, False)
    assert body["error"].startswith("The model's provider is limiting requests")
    assert (body["done"], body["created"]) == (1, 1)
    assert LLMSuggestion.query.count() == 1


def test_the_worker_stops_on_a_daily_limit_and_keeps_its_work(
    client, app, gated, fake_redis, limited, slept
):
    limited.answers = [(200, {}, None), (200, {}, None), PER_DAY]
    body, _ = _run_job(client, app, gated, fake_redis)
    assert slept == []
    assert body["status"] == "provider_limited"
    assert body["daily"] is True
    assert body["retry_after"] == 7200
    assert "daily allowance" in body["error"]
    assert (body["done"], body["created"]) == (2, 2)
    assert LLMSuggestion.query.count() == 2
    assert len(limited.calls) == 3


def test_an_api_request_never_sleeps(client, app, gated, limited, slept):
    limited.answers = [PER_MINUTE]
    dataset_id, task_ids = _dataset(app, tasks=1)
    _ack(client, dataset_id, gated)
    assert _suggest(client, dataset_id, task_ids, gated).status_code == 429
    assert slept == []


def test_a_provider_limited_seed_job_still_orders_what_it_suggested(
    app, fake_redis, limited, slept, monkeypatch
):
    ordered = []
    monkeypatch.setattr(suggest, "_order_seed", lambda dataset, info: ordered.append(info["job_id"]))
    limited.answers = [PER_DAY]
    dataset_id, task_ids = _dataset(app, tasks=1)
    process_llm_annotation_task(
        {
            "job_id": "j1",
            "dataset_id": dataset_id,
            "owner_id": TEST_USER_ID,
            "mode": "shared",
            "task_ids": task_ids,
            "seed_batch_size": 1,
        },
        fake_redis,
    )
    assert ordered == ["j1"]
    assert json.loads(fake_redis.get(llm_job_status_key("j1")))["status"] == "provider_limited"


def test_a_meter_that_is_down_leaves_a_free_shared_model_available(client, app, gated, free_shared):
    class Down:
        def remaining(self, owner_id):
            raise ConnectionError("down")

        def record(self, owner_id, tokens, key):
            raise ConnectionError("down")

    app.config["USAGE_METER"] = Down()
    body = client.get("/api/llm/status", headers=gated).get_json()
    assert (body["free_model"], body["shared_key_available"], body["remaining_tokens"]) == (
        True,
        True,
        None,
    )


# Round 2: paid extras, upstream 429s, no quota, the job's wait, the cache


def test_free_without_paid_fallback_lives_with_is_free_model():
    from legal_ml_core.llm import access, models_list

    assert access.free_without_paid_fallback is models_list.free_without_paid_fallback


@pytest.mark.parametrize(
    "extra_body, daily",
    [
        (None, True),
        ({"models": [FREE, "paid/model"]}, False),
        ({"plugins": [{"id": "web"}]}, False),
        ({"web_search_options": {"search_context_size": "low"}}, False),
    ],
)
def test_a_paid_extras_limit_header_is_not_read_as_daily(monkeypatch, extra_body, daily):
    # A paid fallback's 429 names its own per-minute limit, e.g. 200.
    answer = Limited({"X-RateLimit-Limit": "200"}, _openrouter_body("Rate limit exceeded"))
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: answer)
    with pytest.raises(ProviderRateLimited) as caught:
        llm_client.OpenAICompatibleClient(OPENROUTER, "k", FREE, extra_body).chat(
            [{"role": "user", "content": "hi"}]
        )
    assert caught.value.daily is daily


def _upstream_body(message="Rate limit exceeded", headers=None):
    """An upstream provider's 429, relayed by OpenRouter."""
    body = _openrouter_body(message, headers)
    body["error"]["metadata"].update({"provider_name": "Chutes", "provider_code": 429, "raw": "x"})
    return body


def test_an_upstream_providers_limit_header_is_not_daily():
    body = _upstream_body(headers={"X-RateLimit-Limit": "50"})
    assert parse_rate_limit(Limited(body=body), now=NOW, free=True).daily is False
    in_headers = Limited({"X-RateLimit-Limit": "1000"}, _upstream_body())
    assert parse_rate_limit(in_headers, now=NOW, free=True).daily is False


def test_an_upstream_providers_per_day_name_is_still_daily():
    body = _upstream_body("free-models-per-day", {"X-RateLimit-Limit": "50"})
    assert parse_rate_limit(Limited(body=body), now=NOW, free=True).daily is True


NO_QUOTA = (
    "The LLM service says this key has no quota left. "
    "Check your provider's billing, or pick another model."
)


@pytest.mark.parametrize(
    "error",
    [
        {"code": "insufficient_quota", "message": "You exceeded your current quota"},
        {"type": "insufficient_quota", "message": "You exceeded your current quota"},
        {"code": 429, "message": "x", "metadata": {"provider_code": 429, "error_type": "insufficient_quota"}},
    ],
)
def test_no_quota_left_is_not_a_rate_limit(monkeypatch, error):
    answer = Limited({"Retry-After": "15"}, {"error": error})
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: answer)
    with pytest.raises(llm_client.LLMError) as caught:
        llm_client.OpenAICompatibleClient(OPENROUTER, "k", FREE).chat(
            [{"role": "user", "content": "hi"}]
        )
    assert not isinstance(caught.value, ProviderRateLimited)
    assert str(caught.value) == NO_QUOTA


def test_a_job_with_no_quota_fails_and_keeps_its_work(client, app, gated, fake_redis, limited, slept):
    limited.answers = [(200, {}, None), (429, {}, {"error": {"code": "insufficient_quota"}})]
    body, _ = _run_job(client, app, gated, fake_redis)
    assert (body["status"], body["error"]) == ("failed", NO_QUOTA)
    assert slept == []
    assert LLMSuggestion.query.count() == 1


def test_a_jobs_waiting_is_bounded_in_all(client, app, gated, fake_redis, limited, slept):
    wait = (429, {"Retry-After": "110"}, _openrouter_body())
    # Three waits on the first item (330 s), two on the second (550 s); a sixth
    # would take the job past 600 s, so it stops there.
    limited.answers = [wait, wait, wait, (200, {}, None), wait, wait, wait]
    body, _ = _run_job(client, app, gated, fake_redis)
    assert slept == [110] * 5
    assert sum(slept) <= suggest.MAX_WAIT_SECONDS_PER_JOB == 600
    assert body["status"] == "provider_limited"
    assert (body["retry_after"], body["daily"]) == (110, False)
    assert (body["done"], body["created"]) == (1, 1)
    assert len(limited.calls) == 7


@pytest.mark.parametrize(
    "answer, allowance",
    [
        (b'{"data": {"is_free_tier": false}}', (None, None)),
        (b'{"data": {"is_free_tier": true}}', (50, None)),
        (b'{"data": {"is_free_tier": false, "free_model_daily_requests": {"limit": 1000, "remaining": 7}}}', (1000, 7)),
        (b'{"data": {}}', (50, None)),
    ],
)
def test_the_allowance_for_a_tier(listing, answer, allowance):
    from legal_ml_core.llm.models_list import openrouter_free_allowance

    listing.responses["/key"] = (200, answer)
    assert openrouter_free_allowance(OPENROUTER, SECRET) == allowance


@pytest.mark.parametrize("tasks, over", [(50, False), (51, True)])
def test_an_unknown_allowance_warns_past_the_lower_figure(
    client, app, gated, listing, free_shared, tasks, over
):
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": false}}')
    dataset_id, _ = _dataset(app, tasks=tasks)
    body = _estimate(client, dataset_id, gated)
    assert (body["provider_daily_limit"], body["provider_daily_remaining"]) == (None, None)
    assert body["over_provider_limit"] is over


class BalanceUnread:
    """A meter whose balance read is down, and counts every read."""

    def __init__(self):
        self.reads = 0
        self.records = []

    def remaining(self, owner_id):
        self.reads += 1
        raise ConnectionError("down")

    def record(self, owner_id, tokens, key):
        self.records.append(tokens)


def test_a_free_shared_model_never_reads_the_balance(client, app, gated, provider, free_shared, listing, fake_redis):
    meter = BalanceUnread()
    app.config["USAGE_METER"] = meter
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": true}}')
    dataset_id, task_ids = _dataset(app, tasks=2)

    resp = client.get(f"/api/llm/status?dataset_id={dataset_id}", headers=gated)
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert (body["remaining_tokens"], body["shared_key_available"], body["acknowledged"]) == (
        None,
        True,
        False,
    )

    _ack(client, dataset_id, gated)
    assert _estimate(client, dataset_id, gated)["remaining_tokens"] is None
    assert _suggest(client, dataset_id, task_ids[:1], gated).status_code == 200
    _start_job(client, dataset_id, gated)
    assert meter.reads == 0
    assert meter.records == [100]  # usage is still recorded


def test_status_rolls_back_a_failed_balance_read_before_the_dataset(client, app, gated):
    from legal_ml_core import AnnotationTask

    class Poisoning:
        """Leaves the session needing a rollback, then fails."""

        def remaining(self, owner_id):
            db.session.add(AnnotationTask(dataset_id=None, text=None))
            db.session.flush()

        def record(self, owner_id, tokens, key):
            pass

    app.config["USAGE_METER"] = Poisoning()
    dataset_id, _ = _dataset(app, tasks=1)
    resp = client.get(f"/api/llm/status?dataset_id={dataset_id}", headers=gated)
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert (body["shared_key_available"], body["acknowledged"]) == (False, False)


def test_the_allowance_is_kept_a_minute_per_key(listing, monkeypatch):
    from legal_ml_core.llm import models_list
    from legal_ml_core.llm.models_list import openrouter_free_allowance

    clock = [1000.0]
    monkeypatch.setattr(models_list, "_cache_clock", lambda: clock[0])
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": true}}')

    assert openrouter_free_allowance(OPENROUTER, SECRET) == (50, None)
    clock[0] += 59
    assert openrouter_free_allowance(OPENROUTER, SECRET) == (50, None)
    assert len(listing.calls) == 1
    # Another key is asked for its own.
    openrouter_free_allowance(OPENROUTER, "another-key")
    assert len(listing.calls) == 2
    # The cache holds the key's hash, never the key.
    assert SECRET not in repr(models_list._allowance_cache)
    clock[0] += 2
    listing.responses["/key"] = (
        200,
        b'{"data": {"free_model_daily_requests": {"limit": 1000, "remaining": 3}}}',
    )
    assert openrouter_free_allowance(OPENROUTER, SECRET) == (1000, 3)
    assert len(listing.calls) == 3


def test_a_failed_allowance_lookup_is_not_kept(listing, monkeypatch):
    from legal_ml_core.llm import models_list
    from legal_ml_core.llm.models_list import openrouter_free_allowance

    monkeypatch.setattr(models_list, "_cache_clock", lambda: 1000.0)
    listing.responses["/key"] = (500, b"oops")
    assert openrouter_free_allowance(OPENROUTER, SECRET) == (50, None)
    listing.responses["/key"] = (200, b'{"data": {"is_free_tier": false}}')
    assert openrouter_free_allowance(OPENROUTER, SECRET) == (None, None)
    assert len(listing.calls) == 2
