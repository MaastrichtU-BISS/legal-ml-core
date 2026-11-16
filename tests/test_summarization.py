"""Summarisation, folded in from legal-summarizer-service: the compute
module, the /api/summarize operation, and the worker loop."""
import json

import pytest

from legal_ml_core.compute import summarization
from legal_ml_core.compute.summarization import (
    SummaryRequest,
    chunk_text,
    normalise_model,
    parse_request,
    readability_pass,
)
from legal_ml_core.queue_contract import SUMMARIZATION_QUEUE, summary_status_key
from legal_ml_core.summary_state import SUMMARY_FAILED_MESSAGE, write_summary_status
from legal_ml_core.worker import _summarization_loop
from tests.conftest import TEST_USER_ID, _TEST_CLAIMS

TEXT = " ".join(
    [
        "The licence agreement governs access to marine genetic resources collected in the high seas.",
        "Each party must notify the clearing-house before any sample leaves the collection area.",
        "Benefit sharing is required when the resources are used for commercial purposes.",
        "The agreement was signed in Osaka.",
        "Disputes between the parties are settled by arbitration under the rules of the permanent court.",
        "Monetary benefits are paid into a special fund administered by the secretariat.",
        "Non-monetary benefits include capacity building, technology transfer and access to data.",
        "Users must keep records of every sample, its origin and its subsequent use for twenty years.",
        "The secretariat publishes an annual report on notifications and benefits received.",
        "Either party may terminate the agreement with six months written notice.",
        "Termination does not affect obligations that arose before the termination date.",
        "Confidential business information is protected and not published by the clearing-house.",
    ]
)


def _headers_for(sub):
    claims = dict(_TEST_CLAIMS, sub=sub)
    return {"Authorization": f"Bearer test-token::{json.dumps(claims)}"}


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["lex_rank", "lsa", "text_rank", "luhn", "edmundson"])
def test_every_extractive_algorithm_returns_sentences_from_the_text(model):
    summary = summarization.summarize(SummaryRequest(input_text=TEXT, summary_length="short", model=model))
    assert summary
    assert all(sentence.strip(". ") in TEXT for sentence in summary.split(". "))


def test_summary_length_sets_the_sentence_count():
    def count(length):
        req = SummaryRequest(input_text=TEXT, summary_length=length, readability=0)
        return summarization.summarize(req).count(". ") + 1

    assert count("short") == 3
    assert count("medium") == 6
    assert count(40) == 2  # ceil(40 words / 20)


def test_text_shorter_than_the_target_is_returned_whole():
    short = "One sentence only."
    assert summarization.summarize(SummaryRequest(input_text=short, summary_length="long")) == short


def test_readability_zero_keeps_the_algorithms_top_sentences():
    sentences = ["a b c", "a b c d e f g h i j k l", "a b", "a b c d"]
    assert readability_pass(sentences, 2, 0) == sentences[:2]


def test_readability_one_prefers_median_length_sentences_in_document_order():
    sentences = ["w " * 30, "w " * 10, "w " * 11, "w " * 2, "w " * 10]
    picked = readability_pass(sentences, 2, 1)
    assert [len(s.split()) for s in picked] == [10, 11] or [len(s.split()) for s in picked] == [10, 10]
    assert picked == [s for s in sentences if s in picked]  # original order kept


def test_long_text_is_chunked_for_abstractive_models():
    assert chunk_text("w " * 10) == ["w " * 10]
    chunks = chunk_text(" ".join(["w"] * 1500))
    assert [len(c.split()) for c in chunks] == [700, 700, 100]


def test_abstractive_runs_each_chunk_and_reports_progress(monkeypatch):
    calls = []

    def fake_pipe(chunk, **kwargs):
        calls.append(kwargs)
        return [{"summary_text": f"part{len(calls)}"}]

    monkeypatch.setattr(summarization, "_load_pipeline", lambda model_id: fake_pipe)
    progress = []
    req = SummaryRequest(
        input_text=" ".join(["w"] * 1500), summary_length="long", relevance=0.9, model="hf:facebook/bart-large-cnn"
    )
    assert summarization.summarize(req, on_progress=progress.append) == "part1 part2 part3"
    assert all(c["num_beams"] == 6 for c in calls)
    assert calls[0]["max_length"] == 400 // 3
    assert progress == sorted(progress) and progress[-1] == 95


def test_models_are_normalised_and_unknown_ones_rejected():
    assert normalise_model(None) == "lex_rank"
    assert normalise_model("sumy:lsa") == "lsa"
    assert normalise_model("hf:sshleifer/distilbart-cnn-12-6") == "hf:sshleifer/distilbart-cnn-12-6"
    with pytest.raises(ValueError, match="Unknown model"):
        normalise_model("bert")
    with pytest.raises(ValueError, match="not offered"):
        normalise_model("hf:someone/arbitrary-model")


def test_a_deployment_can_replace_the_abstractive_catalogue():
    config = {"LEGAL_ML_SUMMARIZATION_HF_MODELS": ["org/house-model"]}
    assert normalise_model("hf:org/house-model", config) == "hf:org/house-model"
    with pytest.raises(ValueError):
        normalise_model("hf:facebook/bart-large-cnn", config)


@pytest.mark.parametrize(
    "body, message",
    [
        ({}, "input_text is required"),
        ({"input_text": "   "}, "input_text is required"),
        ({"input_text": "x", "summary_length": "huge"}, "summary_length"),
        ({"input_text": "x", "summary_length": 0}, "summary_length"),
        ({"input_text": "x", "summary_length": True}, "summary_length"),
        ({"input_text": "x", "priority": {"relevance": 2}}, "priority.relevance"),
        ({"input_text": "x", "priority": {"readability": "high"}}, "priority.readability"),
        ({"input_text": "x", "priority": []}, "priority must be an object"),
    ],
)
def test_invalid_requests_are_rejected_with_a_reason(body, message):
    with pytest.raises(ValueError, match=message):
        parse_request(body)


def test_input_longer_than_the_limit_is_rejected():
    with pytest.raises(ValueError, match="longer than 10"):
        parse_request({"input_text": "x" * 11}, {"LEGAL_ML_SUMMARIZATION_MAX_INPUT_CHARS": 10})


# ---------------------------------------------------------------------------
# /api/summarize
# ---------------------------------------------------------------------------

def test_submit_queues_the_job_for_the_worker(client, auth_headers, fake_redis):
    resp = client.post(
        "/api/summarize",
        json={"request_id": "r1", "input_text": TEXT, "summary_length": 50, "model": "sumy:luhn"},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    assert resp.get_json() == {
        "request_id": "r1",
        "status": "queued",
        "progress": 0,
        "summarized_text": None,
        "error": None,
    }
    task = json.loads(fake_redis.rpop(SUMMARIZATION_QUEUE))
    assert task["owner_id"] == TEST_USER_ID
    assert task["model"] == "luhn"
    assert task["summary_length"] == 50
    assert task["priority"] == {"relevance": 0.5, "readability": 0.5}


def test_submit_generates_a_request_id_when_none_is_given(client, auth_headers):
    resp = client.post("/api/summarize", json={"input_text": TEXT}, headers=auth_headers)
    assert resp.status_code == 202
    assert len(resp.get_json()["request_id"]) == 36


def test_submit_rejects_bad_input_and_reused_ids(client, auth_headers):
    assert client.post("/api/summarize", json={"input_text": ""}, headers=auth_headers).status_code == 400
    assert client.post("/api/summarize", json={"input_text": "x", "model": "gpt"}, headers=auth_headers).status_code == 400
    bad_id = client.post("/api/summarize", json={"input_text": "x", "request_id": "a/b"}, headers=auth_headers)
    assert bad_id.status_code == 400
    assert client.post("/api/summarize", json={"input_text": "x", "request_id": "dup"}, headers=auth_headers).status_code == 202
    assert client.post("/api/summarize", json={"input_text": "x", "request_id": "dup"}, headers=auth_headers).status_code == 409


def test_summarize_needs_a_token(client):
    assert client.post("/api/summarize", json={"input_text": TEXT}).status_code == 401
    assert client.get("/api/summarize/r1").status_code == 401
    assert client.get("/api/summarize/models").status_code == 401


def test_status_is_returned_to_its_owner_only(client, auth_headers, fake_redis):
    write_summary_status(fake_redis, "r2", TEST_USER_ID, "complete", 100, summarized_text="done")
    resp = client.get("/api/summarize/r2", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json() == {
        "request_id": "r2",
        "status": "complete",
        "progress": 100,
        "summarized_text": "done",
        "error": None,
    }
    other = client.get("/api/summarize/r2", headers=_headers_for("someone-else"))
    assert other.status_code == 404
    assert client.get("/api/summarize/missing", headers=auth_headers).status_code == 404


def test_a_queued_job_says_when_no_worker_is_running(client, auth_headers, fake_redis):
    write_summary_status(fake_redis, "r3", TEST_USER_ID, "queued", 0)
    body = client.get("/api/summarize/r3", headers=auth_headers).get_json()
    assert body["status_message"] == "Waiting: the ML worker isn't running"


def test_models_lists_the_catalogue(client, auth_headers, app):
    body = client.get("/api/summarize/models", headers=auth_headers).get_json()
    assert body["default"] == "lex_rank"
    assert [m["id"] for m in body["extractive"]] == ["lex_rank", "lsa", "text_rank", "luhn", "edmundson"]
    assert "hf:facebook/bart-large-cnn" in [m["id"] for m in body["abstractive"]]
    assert isinstance(body["abstractive_available"], bool)

    app.config["LEGAL_ML_SUMMARIZATION_HF_MODELS"] = []
    body = client.get("/api/summarize/models", headers=auth_headers).get_json()
    assert body["abstractive"] == [] and body["abstractive_available"] is False


# ---------------------------------------------------------------------------
# worker loop
# ---------------------------------------------------------------------------

def _queue(fake_redis, request_id, **overrides):
    task = dict(
        SummaryRequest(input_text=TEXT, summary_length="short").to_payload(),
        request_id=request_id,
        owner_id=TEST_USER_ID,
        **overrides,
    )
    write_summary_status(fake_redis, request_id, TEST_USER_ID, "queued", 0)
    fake_redis.lpush(SUMMARIZATION_QUEUE, json.dumps(task))


def _run(fake_redis):
    _summarization_loop(fake_redis, stop=lambda: fake_redis.llen(SUMMARIZATION_QUEUE) == 0, sleep=lambda s: None)


def _status(fake_redis, request_id):
    return json.loads(fake_redis.get(summary_status_key(request_id)))


def test_worker_completes_a_queued_summary(fake_redis):
    _queue(fake_redis, "w1")
    _run(fake_redis)
    state = _status(fake_redis, "w1")
    assert state["status"] == "complete" and state["progress"] == 100
    assert state["summarized_text"] and state["owner_id"] == TEST_USER_ID
    assert fake_redis.ttl(summary_status_key("w1")) > 0


def test_worker_reports_a_known_failure_in_plain_words(fake_redis, monkeypatch):
    def boom(req, on_progress=None):
        raise RuntimeError("Abstractive summarisation needs transformers and torch")

    monkeypatch.setattr(summarization, "summarize", boom)
    _queue(fake_redis, "w2")
    _run(fake_redis)
    assert _status(fake_redis, "w2")["error"] == "Abstractive summarisation needs transformers and torch"


def test_worker_hides_unexpected_errors_and_keeps_serving(fake_redis, monkeypatch):
    real = summarization.summarize

    def flaky(req, on_progress=None):
        if req.model == "lsa":
            raise KeyError("internal detail")
        return real(req, on_progress)

    monkeypatch.setattr(summarization, "summarize", flaky)
    fake_redis.lpush(SUMMARIZATION_QUEUE, "not json")
    _queue(fake_redis, "bad", model="lsa")
    _queue(fake_redis, "good")
    _run(fake_redis)
    assert _status(fake_redis, "bad") == {
        "request_id": "bad",
        "owner_id": TEST_USER_ID,
        "status": "error",
        "progress": 100,
        "summarized_text": None,
        "error": SUMMARY_FAILED_MESSAGE,
    }
    assert _status(fake_redis, "good")["status"] == "complete"
