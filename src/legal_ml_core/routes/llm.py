"""LLM assistance API: access, keys, prompts, suggestions and standalone jobs."""
import json
import re
import uuid
from datetime import datetime
from urllib.parse import urlsplit

from flask import Blueprint, current_app, jsonify, request

from ..auth import get_jwt, get_jwt_identity, jwt_required
from ..db import db
from ..llm.access import (
    CredentialError,
    acknowledgement,
    decrypt_key,
    encrypt_key,
    resolve_credential,
    shared_key_allowed,
    validate_byok_url,
)
from ..llm.client import LLMError
from ..llm.config import float_setting, int_setting, shared_endpoint
from ..llm.models_list import MESSAGES as LISTING_MESSAGES, ModelListError, list_chat_models
from ..llm.prompts import (
    PromptRejected,
    check_template,
    parse_scope_verdict,
    scope_messages,
)
from ..llm.suggest import (
    CONFIRMED_SOURCES,
    BudgetExhausted,
    METER_DOWN_MESSAGE,
    call_metered,
    current_prompt,
    estimate_job_tokens,
    remaining_tokens,
    suggest_tasks,
)
from ..models import (
    AnnotationTask,
    Dataset,
    LLMAcknowledgement,
    LLMCredential,
    LLMPrompt,
    LLMSuggestion,
)
from ..queue_contract import LLM_JOB_STATUS_TTL_SECONDS, llm_job_status_key

llm_bp = Blueprint("llm", __name__)

# Single-label only for now: these are the tasks that train a classifier.
SUPPORTED_TASK_TYPES = {
    "classification",
    "document_classification",
    "judgment_prediction",
}


class ApiError(Exception):
    def __init__(self, status, message, code=None, **extra):
        super().__init__(message)
        self.status = status
        self.body = {"error": message, **({"code": code} if code else {}), **extra}


@llm_bp.errorhandler(ApiError)
def _api_error(error):
    return jsonify(error.body), error.status


@llm_bp.errorhandler(CredentialError)
def _credential_error(error):
    return jsonify({"error": str(error), "code": "credential_unusable"}), 400


@llm_bp.errorhandler(LLMError)
def _llm_error(error):
    return jsonify({"error": str(error), "code": "llm_failed"}), 502


@llm_bp.errorhandler(BudgetExhausted)
def _budget_error(error):
    return (
        jsonify(
            {
                "error": "You've reached the monthly limit. Add your own API key to continue.",
                "code": "budget_exhausted",
                "byok_available": True,
            }
        ),
        429,
    )


def _owner():
    return str(get_jwt_identity())


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _dataset(dataset_id):
    """Not-found and not-yours are deliberately indistinguishable."""
    dataset = Dataset.query.filter_by(id=dataset_id, owner_id=_owner()).first()
    if dataset is None:
        raise ApiError(404, "Dataset not found")
    return dataset


def _labelable_dataset(dataset_id):
    dataset = _dataset(dataset_id)
    if dataset.task_type not in SUPPORTED_TASK_TYPES:
        raise ApiError(400, "LLM annotation supports single-label classification only.")
    labels = dataset.labels
    if not isinstance(labels, list) or len(labels) < 2:
        raise ApiError(400, "Define at least two labels for this dataset first.")
    return dataset


def _scope_id(dataset):
    return str(dataset.project_id or dataset.id)


def _credential():
    credential = resolve_credential(_owner(), get_jwt())
    if credential is None:
        raise ApiError(
            403,
            "LLM access isn't available for this account. Add your own API key to use it.",
            code="no_llm_access",
            byok_available=True,
        )
    return credential


def _ready_credential(dataset):
    """A usable credential whose destination this project has acknowledged."""
    credential = _credential()
    if acknowledgement(_owner(), _scope_id(dataset), credential.destination) is None:
        raise ApiError(
            409,
            f"Confirm that this project's text may be sent to {credential.destination}.",
            code="acknowledgement_required",
            destination=credential.destination,
        )
    return credential


def _pending_tasks(dataset, task_ids=None):
    query = AnnotationTask.query.filter(
        AnnotationTask.dataset_id == dataset.id,
        AnnotationTask.status == "pending",
        # An LLM label on a held-out item would grade the model against the LLM.
        AnnotationTask.is_holdout.is_(False),
    )
    if task_ids is not None:
        if not isinstance(task_ids, list):
            raise ApiError(400, "task_ids must be a list.")
        query = query.filter(AnnotationTask.id.in_(task_ids))
    return query.order_by(AnnotationTask.item_index).all()


@llm_bp.route("/status", methods=["GET"])
@jwt_required()
def status():
    owner_id = _owner()
    claims = get_jwt()
    key = LLMCredential.query.filter_by(owner_id=owner_id).first()
    # Whether this deployment offers a shared key at all, apart from whether
    # this account may use it. Nothing about the key itself is returned.
    shared_configured = shared_endpoint() is not None
    shared_allowed = shared_key_allowed(claims) and shared_configured

    credential = None
    try:
        credential = resolve_credential(owner_id, claims)
    except CredentialError as exc:
        current_app.logger.info("stored LLM key unusable: %s", exc)

    body = {
        "mode": credential.mode if credential else "none",
        "destination": credential.destination if credential else None,
        "model": credential.model if credential else None,
        "shared_key_allowed": shared_allowed,
        "shared_key_configured": shared_configured,
        "key": {"base_url": key.base_url, "model": key.model} if key else None,
        "remaining_tokens": None,
        # The most items one POST .../suggestions may ask for.
        "max_sync_tasks": int_setting("LLM_MAX_SYNC_TASKS"),
        # False only when the shared key is the credential but its usage meter
        # could not be asked: it is still the key this account would use, so
        # ``mode`` stays ``shared``, yet nothing can be charged to it right
        # now, and the panel can offer the user's own key instead of an error.
        # Anything that would spend the key keeps failing closed.
        "shared_key_available": True,
        "shared_key_unavailable_reason": None,
    }
    if credential and credential.mode == "shared":
        try:
            body["remaining_tokens"] = remaining_tokens(owner_id)
        except LLMError:
            body["shared_key_available"] = False
            body["shared_key_unavailable_reason"] = METER_DOWN_MESSAGE

    dataset_id = request.args.get("dataset_id")
    if dataset_id:
        dataset = _dataset(dataset_id)
        body["acknowledged"] = bool(
            credential
            and acknowledgement(owner_id, _scope_id(dataset), credential.destination)
        )
    return jsonify(body), 200


@llm_bp.route("/key", methods=["PUT"])
@jwt_required()
def save_key():
    data = _body()
    api_key = data.get("api_key")
    base_url = data.get("base_url")
    model = data.get("model")
    for value in (api_key, base_url, model):
        if not isinstance(value, str) or not value.strip():
            raise ApiError(400, "api_key, base_url and model are required.")
    if len(base_url) > 500 or len(model) > 255:
        raise ApiError(400, "base_url or model is too long.")

    base_url = base_url.strip().rstrip("/")
    validate_byok_url(base_url)
    encrypted = encrypt_key(api_key.strip())

    owner_id = _owner()
    row = LLMCredential.query.filter_by(owner_id=owner_id).first()
    if row is None:
        row = LLMCredential(owner_id=owner_id)
        db.session.add(row)
    row.encrypted_key = encrypted
    row.base_url = base_url
    row.model = model.strip()
    db.session.commit()
    # The key itself is never returned.
    return jsonify({"base_url": row.base_url, "model": row.model}), 200


# Printable ASCII with no spaces: anything else can't go in a header as typed.
API_KEY_PATTERN = re.compile(r"[!-~]{1,512}")
BASE_URL_PATTERN = re.compile(r"[!-~]+")
BAD_KEY = "The API key has characters an API key can't contain."
PLAIN_HTTPS = "The base URL must be a plain https URL."


def _models_request(data):
    """``(base_url, api_key)`` from ``{base_url, api_key}`` or ``{use_saved_key: true}``."""
    if data.get("use_saved_key") is True and set(data) == {"use_saved_key"}:
        row = LLMCredential.query.filter_by(owner_id=_owner()).first()
        if row is None:
            raise ApiError(400, "There is no saved API key.", code="no_saved_key")
        unreadable = None
        try:
            api_key = decrypt_key(row.encrypted_key)
        except CredentialError as exc:
            unreadable = str(exc)
        if unreadable is None and not API_KEY_PATTERN.fullmatch(api_key):
            unreadable = LISTING_MESSAGES["saved_key_unreadable"]
        if unreadable is not None:
            raise ApiError(400, unreadable, code="saved_key_unreadable")
        return row.base_url, api_key

    api_key = data.get("api_key")
    base_url = data.get("base_url")
    if (
        set(data) != {"api_key", "base_url"}
        or not isinstance(api_key, str)
        or not api_key.strip()
        or not isinstance(base_url, str)
        or not base_url.strip()
    ):
        raise ApiError(
            400,
            "Send base_url and api_key, or use_saved_key: true.",
            code="invalid_request",
        )
    if len(base_url) > 500:
        raise ApiError(400, "base_url is too long.", code="invalid_request")
    api_key = api_key.strip()
    if not API_KEY_PATTERN.fullmatch(api_key):
        raise ApiError(400, BAD_KEY, code="invalid_request")
    return base_url.strip().rstrip("/"), api_key


def _check_base_url(base_url):
    """``base_url_refused`` unless it is a plain public https URL with no query."""
    refused = None
    if not BASE_URL_PATTERN.fullmatch(base_url):
        refused = PLAIN_HTTPS
    elif "?" in base_url or "#" in base_url:
        refused = "The base URL can't have a query or fragment."
    else:
        try:
            urlsplit(base_url).port  # a bad port or IPv6 literal raises here
            validate_byok_url(base_url)
        except CredentialError as exc:
            refused = str(exc)
        except ValueError:
            refused = PLAIN_HTTPS
    if refused is not None:
        raise ApiError(400, refused, code="base_url_refused")


@llm_bp.route("/models", methods=["POST"])
@jwt_required()
def list_models():
    """The chat models a provider offers, to check a key or pick a model id.

    The key sent here is never stored. Rate limiting is the host's policy.
    Refusals are raised outside ``except`` blocks, so none carries a library
    exception, which can hold the request headers, as its context.
    """
    base_url, api_key = _models_request(_body())
    _check_base_url(base_url)
    refusal = None
    try:
        models = list_chat_models(base_url, api_key)
    except ModelListError as exc:
        refusal = exc
    if refusal is not None:
        # Neither the key nor the provider's answer is logged, only the outcome.
        current_app.logger.info(
            "model listing refused for %s: %s", urlsplit(base_url).hostname, refusal.code
        )
        status = 502 if refusal.code == "provider_unreachable" else 400
        raise ApiError(status, str(refusal), code=refusal.code)
    return jsonify({"models": models}), 200


@llm_bp.route("/key", methods=["DELETE"])
@jwt_required()
def delete_key():
    LLMCredential.query.filter_by(owner_id=_owner()).delete()
    db.session.commit()
    return jsonify({"deleted": True}), 200


@llm_bp.route("/datasets/<dataset_id>/acknowledgement", methods=["POST"])
@jwt_required()
def acknowledge(dataset_id):
    dataset = _dataset(dataset_id)
    credential = _credential()
    # The client must name what the user saw, so a changed destination can't be
    # acknowledged by a stale screen.
    if _body().get("destination") != credential.destination:
        raise ApiError(
            409,
            "The LLM destination has changed. Please review it again.",
            code="destination_changed",
            destination=credential.destination,
        )

    owner_id = _owner()
    scope_id = _scope_id(dataset)
    row = acknowledgement(owner_id, scope_id, credential.destination)
    if row is None:
        row = LLMAcknowledgement(
            owner_id=owner_id, scope_id=scope_id, destination=credential.destination
        )
        db.session.add(row)
        db.session.commit()
        audit = current_app.config.get("LLM_AUDIT_HOOK")
        if audit is not None:
            audit(
                "llm_acknowledged",
                owner_id,
                scope_id=scope_id,
                destination=credential.destination,
            )
    return jsonify(row.to_dict()), 200


@llm_bp.route("/datasets/<dataset_id>/prompt", methods=["GET"])
@jwt_required()
def get_prompt(dataset_id):
    dataset = _dataset(dataset_id)
    version, template, definitions = current_prompt(dataset)
    return (
        jsonify({"version": version, "template": template, "definitions": definitions}),
        200,
    )


@llm_bp.route("/datasets/<dataset_id>/prompt", methods=["PUT"])
@jwt_required()
def update_prompt(dataset_id):
    dataset = _labelable_dataset(dataset_id)
    data = _body()
    template = data.get("template")
    definitions = data.get("definitions", {})
    try:
        check_template(template, definitions, dataset.labels)
    except PromptRejected as exc:
        raise ApiError(400, str(exc), code="prompt_rejected")

    credential = _ready_credential(dataset)
    verdict = call_metered(
        credential, _owner(), scope_messages(template, definitions, dataset.labels)
    )
    try:
        parse_scope_verdict(verdict)
    except PromptRejected as exc:
        raise ApiError(400, str(exc), code="prompt_rejected")

    version = current_prompt(dataset)[0] + 1
    row = LLMPrompt(
        dataset_id=dataset.id, version=version, template=template, definitions=definitions
    )
    db.session.add(row)
    db.session.commit()
    return jsonify(row.to_dict()), 200


@llm_bp.route("/datasets/<dataset_id>/suggestions", methods=["POST"])
@jwt_required()
def create_suggestions(dataset_id):
    """Suggestions for a current batch, generated while the user waits."""
    dataset = _labelable_dataset(dataset_id)
    task_ids = _body().get("task_ids")
    if not task_ids:
        raise ApiError(400, "task_ids is required.")
    limit = int_setting("LLM_MAX_SYNC_TASKS")
    if not isinstance(task_ids, list) or len(task_ids) > limit:
        raise ApiError(400, f"Send at most {limit} task ids. Use annotate for more.")

    credential = _ready_credential(dataset)
    result = suggest_tasks(dataset, _pending_tasks(dataset, task_ids), credential, _owner())
    return (
        jsonify(
            {
                "suggestions": [s.to_dict() for s in result["created"]],
                "skipped": result["skipped"],
                "invalid": result["invalid"],
            }
        ),
        200,
    )


@llm_bp.route("/datasets/<dataset_id>/suggestions", methods=["GET"])
@jwt_required()
def list_suggestions(dataset_id):
    dataset = _dataset(dataset_id)
    query = LLMSuggestion.query.join(AnnotationTask).filter(
        AnnotationTask.dataset_id == dataset.id, AnnotationTask.is_holdout.is_(False)
    )
    status_filter = request.args.get("status")
    if status_filter:
        query = query.filter(LLMSuggestion.status == status_filter)
    suggestions = query.order_by(LLMSuggestion.created_at).all()
    return jsonify({"suggestions": [s.to_dict() for s in suggestions]}), 200


def _owned_suggestion(suggestion_id):
    suggestion = (
        LLMSuggestion.query.join(AnnotationTask)
        .join(Dataset)
        .filter(LLMSuggestion.id == suggestion_id, Dataset.owner_id == _owner())
        .first()
    )
    if suggestion is None:
        raise ApiError(404, "Suggestion not found")
    return suggestion


@llm_bp.route("/suggestions/<suggestion_id>/confirm", methods=["POST"])
@jwt_required()
def confirm_suggestion(suggestion_id):
    """Accept a suggestion, or pass ``label`` to correct it."""
    suggestion = _owned_suggestion(suggestion_id)
    task = suggestion.task
    if task.is_holdout:
        raise ApiError(409, "This item is held back to check the Model, so it can't take a suggestion.")
    label = _body().get("label", suggestion.label)
    if label not in (task.dataset.labels or []):
        raise ApiError(400, "That label isn't one of this dataset's labels.")

    corrected = label != suggestion.label
    suggestion.status = "corrected" if corrected else "accepted"
    task.annotation_label = label
    # The suggestion's label replaces any Assignment; it is rebuilt from the label.
    task.annotation_payload = None
    task.status = "annotated"
    task.label_source = "human" if corrected else "llm_confirmed"
    task.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"suggestion": suggestion.to_dict(), "task": task.to_dict()}), 200


@llm_bp.route("/suggestions/<suggestion_id>/reject", methods=["POST"])
@jwt_required()
def reject_suggestion(suggestion_id):
    suggestion = _owned_suggestion(suggestion_id)
    suggestion.status = "rejected"
    db.session.commit()
    return jsonify({"suggestion": suggestion.to_dict()}), 200


def _agreement(dataset):
    """How often the latest model and prompt matched human decisions.

    A new model or prompt starts at zero, so bulk accept can only ever rest on
    evidence about the suggestions it would accept.
    """
    latest = (
        LLMSuggestion.query.join(AnnotationTask)
        .filter(AnnotationTask.dataset_id == dataset.id)
        .order_by(LLMSuggestion.created_at.desc())
        .first()
    )
    min_sample = int_setting("LLM_AGREEMENT_MIN_SAMPLE")
    threshold = float_setting("LLM_AGREEMENT_THRESHOLD")
    result = {
        "model": latest.model if latest else None,
        "prompt_version": latest.prompt_version if latest else None,
        "checked": 0,
        "agreed": 0,
        "rate": None,
        "min_sample": min_sample,
        "threshold": threshold,
        "unlocked": False,
    }
    if latest is None:
        return result

    pairs = (
        db.session.query(LLMSuggestion.label, AnnotationTask.annotation_label)
        .join(AnnotationTask)
        .filter(
            AnnotationTask.dataset_id == dataset.id,
            AnnotationTask.status == "annotated",
            AnnotationTask.label_source.in_(CONFIRMED_SOURCES),
            LLMSuggestion.model == latest.model,
            LLMSuggestion.prompt_version == latest.prompt_version,
        )
        .all()
    )
    result["checked"] = len(pairs)
    result["agreed"] = sum(1 for suggested, confirmed in pairs if suggested == confirmed)
    if pairs:
        result["rate"] = round(result["agreed"] / len(pairs), 4)
        result["unlocked"] = len(pairs) >= min_sample and result["rate"] >= threshold
    return result


@llm_bp.route("/datasets/<dataset_id>/agreement", methods=["GET"])
@jwt_required()
def agreement(dataset_id):
    return jsonify(_agreement(_dataset(dataset_id))), 200


@llm_bp.route("/datasets/<dataset_id>/bulk-accept", methods=["POST"])
@jwt_required()
def bulk_accept(dataset_id):
    dataset = _dataset(dataset_id)
    check = _agreement(dataset)
    if not check["unlocked"]:
        raise ApiError(
            403,
            f"Bulk accept needs {check['min_sample']} checked items at "
            f"{check['threshold']:.0%} agreement.",
            code="agreement_not_met",
            agreement=check,
        )

    suggestions = (
        LLMSuggestion.query.join(AnnotationTask)
        .filter(
            AnnotationTask.dataset_id == dataset.id,
            AnnotationTask.status == "pending",
            AnnotationTask.is_holdout.is_(False),
            LLMSuggestion.status == "pending",
            LLMSuggestion.model == check["model"],
            LLMSuggestion.prompt_version == check["prompt_version"],
        )
        .all()
    )
    now = datetime.utcnow()
    for suggestion in suggestions:
        suggestion.status = "bulk_accepted"
        suggestion.bulk_agreement = check
        task = suggestion.task
        task.annotation_label = suggestion.label
        task.status = "annotated"
        task.label_source = "llm_bulk"
        task.updated_at = now
    db.session.commit()
    return jsonify({"accepted": len(suggestions), "agreement": check}), 200


def _estimate(dataset, credential, task_ids):
    tasks = _pending_tasks(dataset, task_ids)
    estimated = estimate_job_tokens(dataset, tasks)
    remaining = remaining_tokens(_owner()) if credential.mode == "shared" else None
    return tasks, {
        "tasks": len(tasks),
        "estimated_tokens": estimated,
        "remaining_tokens": remaining,
        "within_budget": remaining is None or estimated <= remaining,
        "destination": credential.destination,
    }


@llm_bp.route("/datasets/<dataset_id>/annotate/estimate", methods=["POST"])
@jwt_required()
def estimate_annotation(dataset_id):
    dataset = _labelable_dataset(dataset_id)
    credential = _credential()
    return jsonify(_estimate(dataset, credential, _body().get("task_ids"))[1]), 200


@llm_bp.route("/datasets/<dataset_id>/annotate", methods=["POST"])
@jwt_required()
def start_annotation(dataset_id):
    """Standalone annotation of a subset or a whole dataset, on the worker."""
    dataset = _labelable_dataset(dataset_id)
    data = _body()
    if data.get("confirm_estimate") is not True:
        raise ApiError(400, "Review the estimate, then send confirm_estimate: true.")

    credential = _ready_credential(dataset)
    tasks, estimate = _estimate(dataset, credential, data.get("task_ids"))
    if not tasks:
        raise ApiError(400, "There are no pending items to annotate.")
    if not estimate["within_budget"]:
        raise ApiError(
            429,
            "This job is larger than your remaining monthly limit.",
            code="budget_exhausted",
            byok_available=True,
            estimate=estimate,
        )

    job_id = str(uuid.uuid4())
    owner_id = _owner()
    redis_client = current_app.redis_client
    redis_client.setex(
        llm_job_status_key(job_id),
        LLM_JOB_STATUS_TTL_SECONDS,
        json.dumps(
            {"owner_id": owner_id, "status": "queued", "total": len(tasks), "done": 0}
        ),
    )
    redis_client.lpush(
        "llm_queue",
        json.dumps(
            {
                "job_id": job_id,
                "dataset_id": dataset.id,
                "owner_id": owner_id,
                "mode": credential.mode,
                "task_ids": [task.id for task in tasks],
            }
        ),
    )
    return jsonify({"job_id": job_id, "estimate": estimate}), 202


@llm_bp.route("/jobs/<job_id>", methods=["GET"])
@jwt_required()
def job_status(job_id):
    raw = current_app.redis_client.get(llm_job_status_key(job_id))
    status_body = json.loads(raw) if raw else None
    if status_body is None or status_body.get("owner_id") != _owner():
        raise ApiError(404, "Job not found")
    status_body.pop("owner_id")
    return jsonify(status_body), 200
