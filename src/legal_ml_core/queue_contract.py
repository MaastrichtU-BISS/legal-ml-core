"""Typed documentation of the Redis queue message shapes shared between the
Flask API process (producer) and the worker process (consumer). These are
`TypedDict`s for editor/type-checker support, not runtime-validated schemas —
the existing task-processing code already works and this package's job is to
make the previously-implicit dict shapes explicit, not to add a new
validation layer that could reject payloads the working code already handles.

Queue names (Redis lists, `LPUSH`/`BRPOP`):
  training_queue, prediction_queue, explanation_queue, llm_queue, scoring_queue,
  summarization_queue

Result keys (Redis strings/lists, `SETEX`/`LPUSH` + `GET`/`BLPOP`):
  job_status_{job_id}, prediction_result_{prediction_id}, explanation_{explanation_id},
  llm_job_status_{job_id},
  llm_active_job:{owner_id} -- the owner's one queued or running LLM job; set
  NX by the API when it queues one, deleted by the worker on a final status.
  summary_status_{request_id} -- written by the API when a summary is queued and by
  the worker as it runs; carries the owner_id, which the API strips on read.

Control keys (Redis strings, `SETEX` by the API, `EXISTS` by the worker):
  job_cancel_{job_id} -- set by POST /training/stop/<job_id>; the worker checks
  it at pickup, between preprocessing/fit/save stages, and every HF Trainer
  step, and stops the job cooperatively (see compute/tasks.py).

Liveness keys (Redis strings with a TTL, written by the worker; see liveness.py):
  job_heartbeat_{job_id} -- rewritten every heartbeat while the job runs; each
  beat also refreshes job_status_{job_id}'s TTL. Absent for a `running` job
  => the worker died (the API sweep marks it failed).
  legal_ml_worker_alive -- rewritten while any worker process is up. Absent
  => queued jobs report that the training service isn't running.
"""
import json
from typing import Any, List, Optional, TypedDict

from redis import exceptions as redis_exceptions

# Long enough to outlive any queue wait + training run; the worker never deletes
# it, so a late-delivered task for a stopped job still sees the flag.
JOB_CANCEL_TTL_SECONDS = 24 * 3600

# job_status_{job_id} lifetime; heartbeats refresh it so long runs keep it.
JOB_STATUS_TTL_SECONDS = 3600

PREDICTION_RESULT_TTL_SECONDS = 300
BATCH_PREDICTION_RESULT_TTL_SECONDS = 1800
EXPLANATION_RESULT_TTL_SECONDS = 3600
LLM_JOB_STATUS_TTL_SECONDS = 24 * 3600
MODEL_EXPORT_RESULT_TTL_SECONDS = 600

WORKER_LIVENESS_KEY = "legal_ml_worker_alive"


def job_status_key(job_id):
    return f"job_status_{job_id}"


def job_heartbeat_key(job_id):
    return f"job_heartbeat_{job_id}"


def job_cancel_key(job_id):
    return f"job_cancel_{job_id}"


def prediction_result_key(prediction_id):
    return f"prediction_result_{prediction_id}"


def batch_prediction_request_key(batch_id):
    return f"batch_prediction_{batch_id}"


def batch_prediction_result_key(batch_id):
    return f"batch_prediction_result_{batch_id}"


def explanation_result_key(explanation_id):
    return f"explanation_{explanation_id}"


def llm_seed_job_key(dataset_id):
    """The LLM job choosing this dataset's first active learning batch."""
    return f"al_seed_job_{dataset_id}"


def model_export_result_key(export_id):
    """Where the worker answers an export request (BAIGLE #69)."""
    return f"model_export_result_{export_id}"


def llm_job_status_key(job_id):
    return f"llm_job_status_{job_id}"


def llm_active_job_key(owner_id):
    """The one LLM job this owner has queued or running (BAIGLE #225).

    The LLM worker is a single consumer and a job may wait out a provider's
    limit, so one owner's stack of jobs would hold up everyone else's.
    """
    return f"llm_active_job:{owner_id}"


# A job in one of these won't run again; its owner may start another.
LLM_JOB_FINAL_STATUSES = frozenset(
    {"completed", "failed", "budget_exhausted", "provider_limited"}
)


def _llm_job_is_live(redis_client, job_id):
    raw = redis_client.get(llm_job_status_key(job_id))
    if not raw:
        return False
    try:
        status = json.loads(raw).get("status")
    except (TypeError, ValueError, AttributeError):
        return False
    return status not in LLM_JOB_FINAL_STATUSES


def claim_llm_job(redis_client, owner_id, job_id, total):
    """Write a queued job's status and make it the owner's active job.

    Returns None when claimed, else the id of the owner's job still queued or
    running (its queued status is then removed again). A key naming a job whose
    status is gone or final is stale and taken over.
    """
    status_key = llm_job_status_key(job_id)
    redis_client.setex(
        status_key,
        LLM_JOB_STATUS_TTL_SECONDS,
        json.dumps({"owner_id": owner_id, "status": "queued", "total": total, "done": 0}),
    )
    key = llm_active_job_key(owner_id)
    for _ in range(5):
        if redis_client.set(key, job_id, nx=True, ex=LLM_JOB_STATUS_TTL_SECONDS):
            return None
        with redis_client.pipeline() as pipe:
            try:
                pipe.watch(key)
                active = pipe.get(key)
                if active is None:
                    continue
                if _llm_job_is_live(redis_client, active):
                    pipe.reset()
                    redis_client.delete(status_key)
                    return active
                pipe.multi()
                pipe.set(key, job_id, ex=LLM_JOB_STATUS_TTL_SECONDS)
                pipe.execute()
                return None
            except redis_exceptions.WatchError:
                continue
    redis_client.delete(status_key)
    return redis_client.get(key) or ""


def release_llm_job(redis_client, owner_id, job_id):
    """Clear the owner's active job, only if it is still this one."""
    key = llm_active_job_key(owner_id)
    with redis_client.pipeline() as pipe:
        try:
            pipe.watch(key)
            if pipe.get(key) != job_id:
                return
            pipe.multi()
            pipe.delete(key)
            pipe.execute()
        except redis_exceptions.WatchError:
            pass


SUMMARIZATION_QUEUE = "summarization_queue"
SUMMARY_STATUS_TTL_SECONDS = 3600


def summary_status_key(request_id):
    return f"summary_status_{request_id}"


def is_cancel_requested(redis_client, job_id):
    return bool(redis_client.exists(job_cancel_key(job_id)))


class TrainingTaskPayload(TypedDict, total=False):
    job_id: str
    model_id: str
    dataset_id: str
    model_type: str
    parameters: dict
    dataset_path: str
    task_type: str
    preprocessing_options: dict
    test_size: float
    random_state: int
    al_round_id: str  # set on an active learning round's run: rescore after it


class ScoringTaskPayload(TypedDict):
    """Sent once a round's model is saved. The worker rescores the pool."""

    round_id: str
    dataset_id: str
    model_id: str
    strategy: str


class VerifyTaskPayload(TypedDict):
    """Sent by POST /datasets/<id>/verify, on the same queue. The worker grades
    the saved model on the holdout."""

    action: str  # always "verify"
    dataset_id: str
    model_id: str


class JobStatusUpdate(TypedDict, total=False):
    status: str  # "running" | "completed" | "failed" | "cancelled"
    progress: int  # 0-100
    metrics: Optional[dict]
    model_path: Optional[str]
    error: Optional[str]


class PredictionTaskPayload(TypedDict):
    prediction_id: str
    model_id: str
    model_type: str
    texts: List[str]
    user_id: str


class PredictionResult(TypedDict, total=False):
    predictions: List[dict]
    model_type: str
    error: str


class ExplanationTaskPayload(TypedDict):
    explanation_id: str
    model_id: str
    text: str
    method: str  # "lime" | "shap"
    user_id: str


class ExplanationResult(TypedDict, total=False):
    explanation: Any
    text: str
    method: str
    model_id: str
    status: str
    error: str


class LLMAnnotationTaskPayload(TypedDict):
    """Carries the credential *mode*, never a key: the worker reads keys from
    the database or its own environment, so none pass through Redis."""

    job_id: str
    dataset_id: str
    owner_id: str
    mode: str  # "shared" | "byok"
    task_ids: List[str]


class LLMJobStatus(TypedDict, total=False):
    owner_id: str
    # "queued" | "running" | "completed" | "failed" | "budget_exhausted" | "provider_limited"
    status: str
    total: int
    done: int
    created: int
    invalid: int
    error: str
    waiting_seconds: Optional[int]  # set while waiting out a provider's per-minute limit
    retry_after: Optional[int]  # provider_limited: seconds until it may work again, if known
    daily: bool  # provider_limited: the provider's daily allowance is used up


class SummarizationTaskPayload(TypedDict):
    request_id: str
    owner_id: str
    input_text: str
    summary_length: Any  # "short" | "medium" | "long" | int (target words)
    priority: dict  # {"relevance": float, "readability": float}
    model: str  # "lex_rank" | ... | "hf:<hub id>"


class SummaryStatus(TypedDict, total=False):
    request_id: str
    owner_id: str
    status: str  # "queued" | "processing" | "complete" | "error"
    progress: int  # 0-100
    summarized_text: Optional[str]
    error: Optional[str]
