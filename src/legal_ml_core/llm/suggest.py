"""Generating suggestions, for the API (one batch) and the worker (a whole job).

Suggestions are stored apart from labels, so training only ever sees what a human
confirmed or bulk-accepted.
"""

import json
import logging
import time
import uuid

from ..db import db
from ..models import AnnotationTask, Dataset, LLMPrompt, LLMSuggestion
from ..queue_contract import (
    LLM_JOB_FINAL_STATUSES,
    LLM_JOB_STATUS_TTL_SECONDS,
    llm_job_status_key,
    release_llm_job,
)
from .access import CredentialError, credential_for_mode, free_without_paid_fallback
from .client import LLMError, ProviderRateLimited, estimate_tokens
from .config import int_setting
from .metering import usage_meter
from .prompts import (
    DEFAULT_TEMPLATE,
    MAX_EXAMPLES,
    InvalidOutput,
    build_messages,
    parse_suggestion,
)

logger = logging.getLogger(__name__)

CONFIRMED_SOURCES = ("human", "llm_confirmed")
METER_DOWN_MESSAGE = (
    "The shared LLM key is unavailable right now. Your own API key would still work."
)


# A worker job waits out a provider's per-minute limit this many times per
# item, for the provider's Retry-After (or DEFAULT_WAIT_SECONDS), at most
# MAX_WAIT_SECONDS each time (a longer one is given up at once), before it gives up.
MAX_WAITS_PER_ITEM = 3
DEFAULT_WAIT_SECONDS = 20
MAX_WAIT_SECONDS = 120
# The worker is the only LLM consumer, so one job's waits are bounded in all:
# a wait that would take the job's total over this stops it as provider_limited.
MAX_WAIT_SECONDS_PER_JOB = 600

# Replaced in tests, so they don't sleep.
_sleep = time.sleep


class BudgetExhausted(Exception):
    pass


def is_free_credential(credential):
    """A free model with no paid fallback (see ``models_list.free_without_paid_fallback``)."""
    return credential is not None and free_without_paid_fallback(
        credential.base_url, credential.model, getattr(credential, "extra_body", None)
    )


def current_prompt(dataset):
    """``(version, template, definitions)``. Version 0 is the unedited default."""
    row = (
        LLMPrompt.query.filter_by(dataset_id=dataset.id)
        .order_by(LLMPrompt.version.desc())
        .first()
    )
    if row is None:
        return 0, DEFAULT_TEMPLATE, {}
    return row.version, row.template, row.definitions or {}


def confirmed_examples(dataset):
    """Up to ``MAX_EXAMPLES`` confirmed, non-held-out items, balanced across labels.

    Each label gets one example before any label gets a second, in the order the
    Dataset names its labels (labels it no longer names come last). Within a
    label the earliest items come first, and the result keeps item order.
    """
    confirmed = AnnotationTask.query.filter(
        AnnotationTask.dataset_id == dataset.id,
        AnnotationTask.status == "annotated",
        AnnotationTask.label_source.in_(CONFIRMED_SOURCES),
        AnnotationTask.is_holdout.is_(False),
    )
    used = {
        label
        for (label,) in confirmed.with_entities(AnnotationTask.annotation_label).distinct()
        if label is not None
    }
    # Each label once, in the Dataset's order: a label named twice would give its items twice.
    named = [label for label in dict.fromkeys(dataset.labels or []) if label in used]
    labels = named + sorted(used - set(named))
    per_label = [
        confirmed.filter(AnnotationTask.annotation_label == label)
        .order_by(AnnotationTask.item_index)
        .limit(MAX_EXAMPLES)
        .all()
        for label in labels[:MAX_EXAMPLES]  # only the first MAX_EXAMPLES labels can be picked
    ]
    picked = []
    for rank in range(MAX_EXAMPLES):
        for tasks in per_label:
            if rank < len(tasks) and len(picked) < MAX_EXAMPLES:
                picked.append(tasks[rank])
    picked.sort(key=lambda task: task.item_index)
    return [(task.text, task.annotation_label) for task in picked]


def remaining_tokens(owner_id):
    """Fails closed: an unreachable meter blocks the shared key, never BYOK."""
    try:
        return usage_meter().remaining(owner_id)
    except Exception:
        logger.exception("usage meter check failed")
        raise LLMError(METER_DOWN_MESSAGE)


def call_metered(credential, owner_id, messages):
    """One LLM call. The shared key is checked before and metered after.

    A free shared model is metered but not capped: its tokens are recorded, a
    spent budget doesn't stop it, and a meter that can't record is logged only.
    """
    if credential.mode != "shared":
        return credential.client().chat(messages)[0]

    free = is_free_credential(credential)
    if not free:
        remaining = remaining_tokens(owner_id)
        if remaining is not None and remaining <= 0:
            raise BudgetExhausted()
    content, tokens = credential.client().chat(messages)
    try:
        usage_meter().record(owner_id, tokens, str(uuid.uuid4()))
    except Exception:
        logger.exception("usage meter record failed")
        if not free:
            raise LLMError(METER_DOWN_MESSAGE)
        # The failed record may leave the session needing a rollback; without
        # one the suggestion's own commit would fail.
        try:
            db.session.rollback()
        except Exception:
            logger.exception("rollback after a failed usage record failed")
    return content


def _call_waiting(credential, owner_id, messages, on_wait, waited):
    """``call_metered``, waiting out a provider's per-minute limit.

    Waits at most ``MAX_WAITS_PER_ITEM`` times; a daily limit, a wait known to
    be longer than ``MAX_WAIT_SECONDS``, one more 429 after the last wait, or a
    wait that would take the job's total (``waited[0]``, seconds, shared by
    its items) over ``MAX_WAIT_SECONDS_PER_JOB``, is raised at once. A retry
    that is sure to fail would only spend the free model's daily allowance.
    """
    waits = 0
    while True:
        try:
            return call_metered(credential, owner_id, messages)
        except ProviderRateLimited as exc:
            too_long = exc.retry_after is not None and exc.retry_after > MAX_WAIT_SECONDS
            if exc.daily or too_long or waits >= MAX_WAITS_PER_ITEM:
                raise
            seconds = exc.retry_after or DEFAULT_WAIT_SECONDS
            if waited[0] + seconds > MAX_WAIT_SECONDS_PER_JOB:
                raise
            waits += 1
            waited[0] += seconds
            if on_wait:
                on_wait(seconds)
            _sleep(seconds)
            if on_wait:
                on_wait(None)


def estimate_job_tokens(dataset, tasks):
    version, template, definitions = current_prompt(dataset)
    examples = confirmed_examples(dataset)
    per_call = sum(
        estimate_tokens(m["content"])
        for m in build_messages(template, definitions, dataset.labels, examples, "")
    )
    output = int_setting("LLM_MAX_OUTPUT_TOKENS")
    limit = int_setting("LLM_MAX_TEXT_CHARS")
    return sum(
        per_call + output + estimate_tokens(task.text[:limit]) for task in tasks
    )


def suggest_tasks(
    dataset,
    tasks,
    credential,
    owner_id,
    on_progress=None,
    wait_on_rate_limit=False,
    on_wait=None,
):
    """Create suggestions for ``tasks``, one call each, committing as it goes.

    Work already done survives ``BudgetExhausted`` and ``LLMError``, and rerunning
    skips it. Output failing the constraint is discarded and counted as invalid.
    With ``wait_on_rate_limit`` (the worker only: a request never sleeps), a
    provider's per-minute 429 is waited out and the item retried, up to
    ``MAX_WAIT_SECONDS_PER_JOB`` of waiting in all; ``on_wait`` hears the
    seconds before each wait and ``None`` after it.
    """
    version, template, definitions = current_prompt(dataset)
    examples = confirmed_examples(dataset)
    existing = {
        row.task_id
        for row in LLMSuggestion.query.filter(
            LLMSuggestion.task_id.in_([t.id for t in tasks]),
            LLMSuggestion.model == credential.model,
            LLMSuggestion.prompt_version == version,
        )
    }
    result = {"created": [], "skipped": 0, "invalid": 0}
    waited = [0]  # seconds this job has waited on the provider, across its items

    for task in tasks:
        if task.id in existing:
            result["skipped"] += 1
        else:
            messages = build_messages(
                template, definitions, dataset.labels, examples, task.text
            )
            if wait_on_rate_limit:
                content = _call_waiting(credential, owner_id, messages, on_wait, waited)
            else:
                content = call_metered(credential, owner_id, messages)
            try:
                label, rationale = parse_suggestion(content, dataset.labels)
            except InvalidOutput as exc:
                logger.info("LLM output discarded for task %s: %s", task.id, exc)
                result["invalid"] += 1
            else:
                suggestion = LLMSuggestion(
                    task_id=task.id,
                    model=credential.model,
                    prompt_version=version,
                    label=label,
                    rationale=rationale,
                )
                db.session.add(suggestion)
                db.session.commit()
                result["created"].append(suggestion)
        if on_progress:
            on_progress(result)
    return result


def _set_status(redis_client, job_id, **fields):
    key = llm_job_status_key(job_id)
    raw = redis_client.get(key)
    status = json.loads(raw) if raw else {}
    status.update(fields)
    redis_client.setex(key, LLM_JOB_STATUS_TTL_SECONDS, json.dumps(status))
    # A final status frees the owner to start another job.
    if status.get("status") in LLM_JOB_FINAL_STATUSES and status.get("owner_id"):
        release_llm_job(redis_client, status["owner_id"], job_id)


def process_llm_annotation_task(task_info, redis_client):
    """Worker entry point for a standalone annotation job."""
    job_id = task_info["job_id"]
    owner_id = task_info["owner_id"]
    dataset = Dataset.query.filter_by(
        id=task_info["dataset_id"], owner_id=owner_id
    ).first()
    if dataset is None:
        _set_status(redis_client, job_id, status="failed", error="Dataset not found.")
        return

    tasks = (
        AnnotationTask.query.filter(
            AnnotationTask.dataset_id == dataset.id,
            AnnotationTask.id.in_(task_info["task_ids"]),
            AnnotationTask.status == "pending",
        )
        .order_by(AnnotationTask.item_index)
        .all()
    )
    _set_status(redis_client, job_id, status="running", total=len(tasks), done=0)

    def progress(result):
        _set_status(
            redis_client,
            job_id,
            done=len(result["created"]) + result["skipped"] + result["invalid"],
            created=len(result["created"]),
            invalid=result["invalid"],
        )

    def waiting(seconds):
        _set_status(redis_client, job_id, waiting_seconds=seconds)

    try:
        credential = credential_for_mode(owner_id, task_info["mode"])
        if credential is None:
            raise CredentialError("The LLM key for this job is no longer available.")
        suggest_tasks(
            dataset,
            tasks,
            credential,
            owner_id,
            on_progress=progress,
            wait_on_rate_limit=True,
            on_wait=waiting,
        )
    except BudgetExhausted:
        _order_seed(dataset, task_info)
        _set_status(
            redis_client,
            job_id,
            status="budget_exhausted",
            error="The monthly limit was reached. Add your own API key to continue.",
        )
    except ProviderRateLimited as exc:
        # The work already done is committed and stays; done and created stay too.
        _order_seed(dataset, task_info)
        _set_status(
            redis_client,
            job_id,
            status="provider_limited",
            error=str(exc),
            retry_after=exc.retry_after,
            daily=exc.daily,
            waiting_seconds=None,
        )
    except (LLMError, CredentialError) as exc:
        _set_status(redis_client, job_id, status="failed", error=str(exc))
    else:
        _order_seed(dataset, task_info)
        _set_status(redis_client, job_id, status="completed")


def _order_seed(dataset, task_info):
    """An active learning seed job orders the first batch from what it suggested."""
    if not task_info.get("seed_batch_size"):
        return
    from ..active_learning.loop import order_llm_seed

    order_llm_seed(dataset.id, task_info["task_ids"], task_info["seed_batch_size"])
    db.session.commit()


def fail_llm_job(raw, redis_client):
    try:
        job_id = json.loads(raw).get("job_id")
    except Exception:
        job_id = None
    if job_id:
        _set_status(
            redis_client, job_id, status="failed", error="The annotation job failed."
        )
