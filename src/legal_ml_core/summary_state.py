"""Summary job state in Redis, shared by the API and the worker.

A summary job has no database row: it is short-lived and its result is
returned to the caller, not stored. Its state lives under
`summary_status_{request_id}` with a TTL (queue_contract), refreshed on every
write. The owner is stored with it so one caller cannot read another's job.
"""
import json

from .queue_contract import SUMMARY_STATUS_TTL_SECONDS, summary_status_key

TERMINAL_SUMMARY_STATUSES = ("complete", "error")

# Shown for an unexpected failure; the cause goes to the worker log only.
SUMMARY_FAILED_MESSAGE = "Summarisation failed. Try again, or pick another model."


def write_summary_status(redis_client, request_id, owner_id, status, progress, summarized_text=None, error=None):
    state = {
        "request_id": request_id,
        "owner_id": owner_id,
        "status": status,
        "progress": progress,
        "summarized_text": summarized_text,
        "error": error,
    }
    redis_client.setex(summary_status_key(request_id), SUMMARY_STATUS_TTL_SECONDS, json.dumps(state))
    return state


def read_summary_status(redis_client, request_id):
    raw = redis_client.get(summary_status_key(request_id))
    return json.loads(raw) if raw else None


def public_view(state):
    """The status as returned to the caller: everything but the owner."""
    return {k: v for k, v in state.items() if k != "owner_id"}
