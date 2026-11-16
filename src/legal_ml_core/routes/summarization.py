"""Summarisation operation: POST /api/summarize, GET /api/summarize/<id>,
GET /api/summarize/models.

Jobs are queued on `summarization_queue` and run by the worker
(worker._summarization_loop). Clients poll for the result; there is no SSE
stream, because every caller reaches this service through the platform's
request/response proxy.
"""
import json
import logging
import re
import uuid

from flask import Blueprint, current_app, jsonify, request

from ..auth import get_jwt_identity, jwt_required
from ..compute.summarization import (
    DEFAULT_MODEL,
    SummaryValidationError,
    abstractive_available,
    allowed_models,
    parse_request,
)
from ..job_state import DEFAULT_STALE_SECONDS, worker_is_alive
from ..queue_contract import SUMMARIZATION_QUEUE
from ..summary_state import public_view, read_summary_status, write_summary_status

logger = logging.getLogger(__name__)

summarization_bp = Blueprint("summarization", __name__)

SUMMARY_WORKER_DOWN_MESSAGE = "Waiting: the ML worker isn't running"

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _not_found():
    return jsonify({"error": "Summary not found"}), 404


@summarization_bp.route("", methods=["POST"], strict_slashes=False)
@jwt_required()
def submit_summary():
    owner_id = get_jwt_identity()
    body = request.get_json(silent=True)
    try:
        summary_request = parse_request(body, current_app.config)
    except SummaryValidationError as e:
        return jsonify({"error": str(e), "field": e.field}), 400

    request_id = body.get("request_id") or str(uuid.uuid4())
    # "models" is GET /api/summarize/models, so it can never be a job id.
    if not isinstance(request_id, str) or not _REQUEST_ID.match(request_id) or request_id == "models":
        return jsonify({"error": "request_id must be 1-128 letters, digits or _ . : - and not 'models'", "field": "request_id"}), 400

    redis_client = current_app.redis_client
    if read_summary_status(redis_client, request_id) is not None:
        return jsonify({"error": "request_id is already in use"}), 409

    state = write_summary_status(redis_client, request_id, owner_id, "queued", 0)
    payload = dict(summary_request.to_payload(), request_id=request_id, owner_id=owner_id)
    redis_client.lpush(SUMMARIZATION_QUEUE, json.dumps(payload))
    logger.info("summary %s (owner %s) queued, model %s", request_id, owner_id, summary_request.model)
    return jsonify(public_view(state)), 202


@summarization_bp.route("/models", methods=["GET"])
@jwt_required()
def list_models():
    catalogue = allowed_models(current_app.config)
    return jsonify(
        {
            "default": DEFAULT_MODEL,
            "abstractive_available": abstractive_available() and bool(catalogue["abstractive"]),
            **catalogue,
        }
    )


@summarization_bp.route("/<request_id>", methods=["GET"])
@jwt_required()
def get_summary(request_id):
    state = read_summary_status(current_app.redis_client, request_id)
    # Another owner's job answers exactly like a missing one.
    if state is None or state.get("owner_id") != get_jwt_identity():
        return _not_found()
    view = public_view(state)
    if state["status"] == "queued":
        stale_after = current_app.config.get("LEGAL_ML_HEARTBEAT_STALE_SECONDS", DEFAULT_STALE_SECONDS)
        if not worker_is_alive(current_app.redis_client, stale_after):
            view["status_message"] = SUMMARY_WORKER_DOWN_MESSAGE
    return jsonify(view)
