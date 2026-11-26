import json
import logging
import os
import time
from datetime import datetime

import joblib
import pandas as pd
import redis as redis_lib
from flask import current_app

from ..active_learning.scoring import queue_pool_scoring, verify_finished_model
from ..annotation_format import SPAN_TYPES
from ..annotation_queue import (
    apply_saved_annotations,
    ensure_dataset_tasks,
    ner_examples,
    summarization_examples,
)
from ..db import db
from ..job_state import ACTIVE_JOB_STATUSES, COULD_NOT_START_MESSAGE, transition
from ..metric_groups import metric_group_for
from ..models import Dataset, Model, TrainingJob, TrainingMetricPoint
from ..monitoring import FilteredMetricLogger, run_monitoring
from ..liveness import DEFAULT_TIMINGS, Heartbeat
from ..queue_contract import JOB_STATUS_TTL_SECONDS, is_cancel_requested, job_status_key
from ..storage import get_storage_backend
from ..system_metrics import log_system_readings
from .explain import generate_lime_explanation, generate_shap_explanation
from .trainer import PREDICTION_WINDOW_BATCH, MLModelTrainer, TrainingCancelled

logger = logging.getLogger(__name__)

# Redis being unreachable (vs. a bad command): reported as a retryable failure.
REDIS_DOWN_ERRORS = (redis_lib.ConnectionError, redis_lib.TimeoutError)

# Model types that fine-tune a token classifier instead of a text classifier.
TOKEN_CLASSIFICATION_TYPES = ("legalbert_ner", "distilbert_ner")

# Model types that fine-tune a text-to-text model instead of a classifier.
SEQ2SEQ_TYPES = ("t5_small",)


def merge_word_spans(text, word_ids, offsets, labels, scores):
    """Merge token predictions into whole-word entity spans, in character
    offsets into ``text``.

    Every token carries its word's number, and the number repeats across the
    overlapping windows an over-long text is predicted in; so a word's label is
    its first sub-token's — from the first window that saw the word — while the
    span reaches the end of the word's last sub-token, wherever a later window
    saw the rest of it. A continuation piece's own label is ignored. A ``B-``
    label starts a span; an ``I-`` continues an open span of its type and starts
    one when none is open; ``O`` closes.
    """
    words = {}  # word id -> [label, score, start, end], in first-seen (text) order
    for word_id, (start, end), label, score in zip(word_ids, offsets, labels, scores):
        if word_id is None or start == end:  # special and padding tokens
            continue
        if word_id in words:
            words[word_id][3] = max(words[word_id][3], end)
        else:
            words[word_id] = [label, score, start, end]

    spans = []
    current = None
    for label, score, start, end in words.values():
        if label == "O":
            current = None
            continue
        prefix, _, kind = label.partition("-")
        if prefix == "B" or current is None or current["type"] != kind:
            current = {"start": start, "end": end, "type": kind, "scores": [score]}
            spans.append(current)
        else:
            current["end"] = max(current["end"], end)
            current["scores"].append(score)
    return spans


class MetricLogger:
    """Appends TrainingMetricPoint rows for one training job.

    `step` is a per-instance monotonically increasing polling cursor (used only
    by the `/training/<job_id>/metrics?since_step=` endpoint) — it is NOT the
    algorithm's own notion of step/epoch. The semantic x-axes for charts are
    `training_step` and `epoch`, which are populated for the HuggingFace Trainer
    path (via TrainingMetricCallback) and left None for the classical sklearn
    algorithms' coarse phase markers.

    `group` is one of metric_groups.METRIC_GROUPS; left out, it is derived from
    the name.

    Points are buffered and written in one commit once `flush_every_points`
    are waiting or the oldest has waited `flush_every_seconds` (checked when a
    point is logged), and at once for a stage marker. Callers `flush()` when a
    run ends. A failed write keeps the batch, so the cursor has no gaps.
    """

    def __init__(
        self, training_job_id, flush_every_points=50, flush_every_seconds=2.0, clock=time.monotonic
    ):
        self.training_job_id = training_job_id
        self._counter = 0
        self._flush_every_points = flush_every_points
        self._flush_every_seconds = flush_every_seconds
        self._clock = clock
        self._buffer = []
        self._oldest_buffered_at = None

    def log(self, metric_name, value, epoch=None, training_step=None, group=None):
        step = self._counter
        self._counter += 1
        if not self._buffer:
            self._oldest_buffered_at = self._clock()
        group = group or metric_group_for(metric_name)
        self._buffer.append(
            dict(
                training_job_id=self.training_job_id,
                step=step,
                epoch=epoch,
                training_step=training_step,
                metric_name=metric_name,
                metric_group=group,
                value=float(value),
            )
        )
        if (
            group == "stage"
            or len(self._buffer) >= self._flush_every_points
            or self._clock() - self._oldest_buffered_at >= self._flush_every_seconds
        ):
            self.flush()
        return step

    def flush(self):
        """Write the buffered points in one commit."""
        if not self._buffer:
            return
        db.session.add_all(TrainingMetricPoint(**point) for point in self._buffer)
        db.session.commit()
        self._buffer = []


def _mark_running(job_id):
    """Pickup: pending -> running + started_at in the DB (OV-E3)."""
    return transition(
        job_id,
        ACTIVE_JOB_STATUSES,
        "running",
        model_status="training",
        started_at=datetime.utcnow(),
    )


def _mark_cancelled(job_id, redis_client):
    """Settle a stopped job as cancelled. /stop normally already wrote the DB
    row; this guarded write only covers the flag-landed-first window."""
    transition(
        job_id,
        ACTIVE_JOB_STATUSES,
        "cancelled",
        model_status="failed",  # same as /stop
        completed_at=datetime.utcnow(),
    )
    _write_live(redis_client, job_id, {"status": "cancelled"}, JOB_STATUS_TTL_SECONDS)


def _write_live(redis_client, job_id, payload, ttl_seconds):
    """Best-effort live status for the API (A2: it only refines progress). A
    Redis failure here never fails the job."""
    try:
        redis_client.psetex(
            job_status_key(job_id), max(1, int(ttl_seconds * 1000)), json.dumps(payload)
        )
    except Exception as e:
        logger.warning("training job %s: live status not written: %s", job_id, e)


def _job_left_active_in_db(job_id):
    """True once the DB row is no longer pending/running (cancelled by /stop,
    swept as failed, or gone). Reads the column, not the ORM identity map, so
    another process's commit is seen."""
    status = db.session.query(TrainingJob.status).filter(TrainingJob.id == job_id).scalar()
    return status not in ACTIVE_JOB_STATUSES


def _with_kept_group_names(model, job_id, metrics):
    """Fresh metrics, plus the group names the user gave the last run when this
    run used the same settings. Group numbers do not carry over between settings,
    so changed settings drop them.

    Settings compare with the catalogue defaults filled in, so an older snapshot
    of only some settings matches the form's full copy of the same choices. A
    run from before snapshots has no record of its settings, and the Model may
    have been edited since, so its names are dropped too: names on the wrong
    groups are worse than lost names. Monitoring settings do not count."""
    # Here, not at the top: routes.models imports the training code's siblings.
    from ..routes.models import effective_parameters

    names = (model.metrics or {}).get("display_labels")
    if not names:
        return metrics
    job = db.session.get(TrainingJob, job_id)
    previous = (
        TrainingJob.query.filter(
            TrainingJob.model_id == model.id,
            TrainingJob.status == "completed",
            TrainingJob.id != job_id,
        )
        .order_by(TrainingJob.completed_at.desc())
        .first()
    )
    if job is None or previous is None or previous.parameters is None:
        return metrics
    if _learning_settings(effective_parameters(model.model_type, job.parameters)) != _learning_settings(
        effective_parameters(model.model_type, previous.parameters)
    ):
        return metrics
    return {**metrics, "display_labels": names}


def _learning_settings(parameters):
    """The settings that change what a run learns. Monitoring only changes what
    is watched, and runs from before it was recorded have none."""
    return {name: value for name, value in parameters.items() if name != "monitoring"}


def _monitoring(task_info):
    """The run's monitoring, as the API recorded it. A message queued before
    monitoring settings existed gets this worker's defaults."""
    if task_info.get("monitoring"):
        return task_info["monitoring"]
    return run_monitoring(current_app.config, task_info.get("parameters"))


def _chosen_parameters(task_info, monitoring):
    """The flat parameters the trainer reads: the Model's choices, nested under
    "parameters" in the queue message. The server's own values go last, so a
    stored parameter can never replace them."""
    parameters = dict(task_info.get("parameters") or {})
    parameters.setdefault("random_state", task_info.get("random_state", 42))
    parameters.setdefault("test_size", task_info.get("test_size", 0.2))
    parameters["model_id"] = task_info["model_id"]
    parameters["monitoring"] = monitoring
    return parameters


def process_training_task(
    task_info,
    redis_client,
    model_storage_path,
    trainer_factory=MLModelTrainer,
    liveness=DEFAULT_TIMINGS,
    heartbeat_factory=Heartbeat,
    scoring_queue="scoring_queue",
):
    """Runs one training task. Must be called with a Flask app context pushed
    (db.session/Model.query need it) — see legal_ml_core.worker.run_ml_worker.

    This replaces the original ml_worker's raw `psycopg2` UPDATE statements
    against the `models`/`training_jobs` tables with normal SQLAlchemy writes
    through the same ORM classes the API process uses — the two processes now
    share one schema representation instead of two independent ones.

    While the job runs, a heartbeat thread (`heartbeat_factory`, see
    liveness.py) beats every `liveness.heartbeat_interval` seconds; if this
    process dies, the beats stop and the API sweep marks the job failed (R5).
    """
    job_id = task_info["job_id"]
    model_id = task_info["model_id"]
    monitoring = _monitoring(task_info)
    metric_logger = FilteredMetricLogger(
        MetricLogger(
            job_id,
            flush_every_points=monitoring["flush_every_points"],
            flush_every_seconds=monitoring["flush_every_seconds"],
        ),
        monitoring,
    )

    def cancel_requested():
        """Stop check: the Redis flag (cheap, set by /stop), else the DB row,
        which /stop writes first and which also covers a flag that could not
        be set or read."""
        try:
            if is_cancel_requested(redis_client, job_id):
                return True
        except Exception as e:
            logger.warning("training job %s: cancel flag unreadable, using DB: %s", job_id, e)
        return _job_left_active_in_db(job_id)

    def checkpoint():
        """Between-stage check. Raises TrainingCancelled after a Stop.

        Liveness is not checked here: the heartbeat runs on its own thread
        because the stages between checkpoints block for minutes.
        """
        if cancel_requested():
            raise TrainingCancelled(job_id)

    def set_live(payload):
        _write_live(redis_client, job_id, payload, liveness.status_ttl)

    def reached(stage):
        """Log a stage marker, then the resource readings there when the run
        takes them (BAIGLE #94)."""
        metric_logger.log(stage, 1.0)
        if monitoring.get("system_metrics"):
            log_system_readings(metric_logger)

    model_path = None
    heartbeat = None
    try:
        logger.info("training job %s: processing model %s", job_id, model_id)

        if cancel_requested():
            _mark_cancelled(job_id, redis_client)
            logger.info("training job %s was stopped before pickup; skipping", job_id)
            return
        if not _mark_running(job_id):
            logger.info("training job %s is no longer active; skipping", job_id)
            return
        heartbeat = heartbeat_factory(redis_client, job_id, liveness).start()

        set_live({"status": "running", "progress": 0})
        reached("preprocessing_started")

        # Load dataset
        dataset = db.session.get(Dataset, task_info["dataset_id"])

        # Initialize trainer
        trainer = trainer_factory(
            redis_client=redis_client,
            metric_logger=metric_logger,
            cancel_check=cancel_requested,
        )

        # Preprocess data. NER text is never cleaned: cleaning would move every
        # span's character offsets. A span task reads no file either: its
        # examples are the saved spans, and the CSV holds nothing it learns from.
        # A summarization task reads the file itself, verbatim (NA, null and
        # 00123 stay text) and uncleaned (a summary is text), by column name.
        span_task = (
            task_info["task_type"] in SPAN_TYPES
            and task_info["model_type"] in TOKEN_CLASSIFICATION_TYPES
        )
        seq2seq_task = (
            task_info["task_type"] == "summarization"
            and task_info["model_type"] in SEQ2SEQ_TYPES
        )
        if span_task:
            # The worker may write: an imported Dataset gets its tasks here,
            # before its spans are read.
            ensure_dataset_tasks(dataset)
            examples = ner_examples(dataset)
            texts, labels, label_names = None, None, None
        elif seq2seq_task:
            examples = summarization_examples(task_info["dataset_path"], dataset)
            texts, labels, label_names = None, None, None
        else:
            df = pd.read_csv(task_info["dataset_path"])
            df = apply_saved_annotations(df, dataset)
            texts, labels, label_names = trainer.preprocess_data(
                df, task_info["task_type"], task_info.get("preprocessing_options", {})
            )
        checkpoint()

        set_live({"status": "running", "progress": 25})
        reached("preprocessing_done")
        reached("training_started")

        # Train model based on task type
        parameters = _chosen_parameters(task_info, monitoring)
        if span_task:
            parameters["output_dir"] = get_storage_backend().get_transformer_output_dir(
                model_storage_path, model_id
            )
            metrics = trainer.train_token_classification(
                examples, task_info["model_type"], parameters
            )
        elif seq2seq_task:
            parameters["output_dir"] = get_storage_backend().get_transformer_output_dir(
                model_storage_path, model_id
            )
            # Checks for a Stop between generation batches and raises
            # TrainingCancelled itself, skipping the scoring pass.
            metrics = trainer.train_seq2seq(examples, task_info["model_type"], parameters)
        elif task_info["model_type"] in ["legalbert", "roberta"]:
            parameters["output_dir"] = get_storage_backend().get_transformer_output_dir(
                model_storage_path, model_id
            )
            # Checks for a Stop every HF step and raises TrainingCancelled
            # itself, skipping evaluate/predict.
            metrics = trainer.train_transformer_model(
                texts, labels, task_info["model_type"], parameters
            )
        elif task_info["task_type"] in [
            "classification",
            "document_classification",
            "judgment_prediction",
        ]:
            metrics = trainer.train_classification_model(
                texts, labels, task_info["model_type"], parameters
            )
        elif task_info["task_type"] in ["topic_modeling", "clustering"]:
            if task_info["model_type"] in ["kmeans", "hierarchical", "dbscan"]:
                metrics = trainer.train_clustering_model(
                    texts, task_info["model_type"], parameters
                )
            else:
                metrics = trainer.train_topic_modeling(
                    texts, task_info["model_type"], parameters
                )
        else:
            raise ValueError(f"Unsupported task type: {task_info['task_type']}")
        checkpoint()

        set_live({"status": "running", "progress": 75})
        reached("training_done")

        # Save model
        model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
        trainer.save_model(model_path)
        metrics = verify_finished_model(db.session.get(Dataset, task_info["dataset_id"]), model_path, metrics)

        # Guarded final write (D4): only a still-active job becomes completed.
        # If Stop landed at any point up to here, the write misses and the
        # model file just written is removed.
        if not transition(
            job_id,
            ACTIVE_JOB_STATUSES,
            "completed",
            model_status="completed",
            metrics=metrics,
            progress=100,
            completed_at=datetime.utcnow(),
        ):
            raise TrainingCancelled(job_id)

        model = db.session.get(Model, model_id)
        if model:
            model.metrics = _with_kept_group_names(model, job_id, metrics)
            model.file_path = model_path
        db.session.commit()
        reached("model_saved")
        _call_swallowing_errors_with_rollback(metric_logger.flush)

        set_live(
            {
                "status": "completed",
                "progress": 100,
                "metrics": metrics,
                "model_path": model_path,
            }
        )
        logger.info("training job %s: completed model %s", job_id, model_id)

        if task_info.get("al_round_id"):
            queue_pool_scoring(task_info, redis_client, scoring_queue)

    except Exception as e:
        db.session.rollback()
        # A Stop (or any error raised while a Stop is in flight) settles as
        # cancelled, never failed (OV-E4).
        if isinstance(e, TrainingCancelled) or _call_swallowing_errors_with_rollback(
            cancel_requested, default=False
        ):
            logger.info("training job %s cancelled", job_id)
            _discard_model_file(model_path)
            _mark_cancelled(job_id, redis_client)
            _call_swallowing_errors_with_rollback(
                lambda: metric_logger.log("training_cancelled", 1.0)
            )
            return

        _discard_model_file(model_path)
        _call_swallowing_errors_with_rollback(lambda: metric_logger.log("training_failed", 1.0))
        _call_swallowing_errors_with_rollback(metric_logger.flush)
        if isinstance(e, REDIS_DOWN_ERRORS):
            # Infrastructure, not the user's data: never show raw Redis text.
            error_message = COULD_NOT_START_MESSAGE
            logger.error(
                "training job %s (owner %s): Redis unreachable during training: %s",
                job_id,
                _owner_of(job_id),
                e,
            )
        else:
            # Persist the reason so the UI can explain the failure instead of
            # showing a generic "Training failed".
            error_message = str(e)
            logger.warning("training job %s failed: %s", job_id, e, exc_info=True)
        # Guarded like the success write: a job that went terminal meanwhile
        # (e.g. cancelled by /stop) keeps its state.
        if transition(
            job_id,
            ACTIVE_JOB_STATUSES,
            "failed",
            model_status="failed",
            error_message=error_message,
            completed_at=datetime.utcnow(),
        ):
            set_live({"status": "failed", "error": error_message})
    finally:
        # Any points still batched, however the run ended (a stop included).
        _call_swallowing_errors_with_rollback(metric_logger.flush)
        if heartbeat is not None:
            heartbeat.stop()


def _owner_of(job_id):
    job = _call_swallowing_errors_with_rollback(lambda: db.session.get(TrainingJob, job_id))
    return job.owner_id if job else None


def _call_swallowing_errors_with_rollback(fn, default=None):
    """Run `fn`; on any error roll the session back and return `default`."""
    try:
        return fn()
    except Exception:
        db.session.rollback()
        return default


def _discard_model_file(model_path):
    if model_path and os.path.exists(model_path):
        try:
            os.remove(model_path)
        except OSError:
            pass


def process_prediction_task(task_info, model_storage_path):
    """Runs one prediction task and returns the result dict for the caller to
    push into Redis. Does not touch the database."""
    prediction_id = task_info["prediction_id"]
    model_id = task_info["model_id"]
    model_type = task_info["model_type"]
    texts = task_info["texts"]  # always a list

    logger.info("prediction for model %s (%d samples)", model_id, len(texts))

    model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model_data = joblib.load(model_path)
    preprocessor = model_data.get("preprocessor")
    vectorizer = model_data.get("vectorizer")
    clf = model_data.get("model")
    label_encoder = model_data.get("label_encoder")

    cleaned = [preprocessor.clean_text(t) if preprocessor else t for t in texts]
    predictions = []

    if model_type in ("svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression"):
        X = vectorizer.transform(cleaned)
        raw_preds = clf.predict(X)
        has_proba = hasattr(clf, "predict_proba")
        probas = clf.predict_proba(X) if has_proba else None
        classes = label_encoder.classes_ if label_encoder else None

        for i, pred in enumerate(raw_preds):
            label = label_encoder.inverse_transform([pred])[0] if label_encoder else str(pred)
            entry = {"text": texts[i], "prediction": label}
            if has_proba and classes is not None:
                entry["confidence"] = round(float(probas[i][pred]), 4)
                entry["all_confidences"] = {
                    cls: round(float(probas[i][j]), 4)
                    for j, cls in enumerate(classes)
                }
            predictions.append(entry)

    elif model_type in ("kmeans", "hierarchical", "dbscan"):
        X = vectorizer.transform(cleaned)
        if hasattr(X, "toarray"):
            X = X.toarray()
        cluster_ids = clf.predict(X) if hasattr(clf, "predict") else clf.fit_predict(X)
        for i, cid in enumerate(cluster_ids):
            predictions.append({
                "text": texts[i],
                "prediction": f"Cluster {int(cid)}",
                "cluster_id": int(cid),
            })

    elif model_type in ("lda", "nmf"):
        X = vectorizer.transform(cleaned)
        topic_weights = clf.transform(X)
        for i, weights in enumerate(topic_weights):
            dominant = int(weights.argmax())
            predictions.append({
                "text": texts[i],
                "prediction": f"Topic {dominant}",
                "dominant_topic": dominant,
                "topic_weights": [round(float(w), 4) for w in weights],
            })

    elif model_type in TOKEN_CLASSIFICATION_TYPES:
        import torch

        tokenizer = model_data.get("tokenizer")
        label_list = model_data.get("label_list")
        if tokenizer is None or not label_list:
            raise ValueError("Tokenizer or entity labels not found in saved model")
        max_length = int(model_data.get("max_length") or 256)

        clf.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device)

        for text in texts:
            # An over-long text is predicted in overlapping windows of the
            # training-time max_length, not cut: the stride keeps every word
            # whole in at least one window, so nothing past a cut is lost.
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                stride=max(1, max_length // 4),
                padding=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            offsets = encoding.pop("offset_mapping")
            encoding.pop("overflow_to_sample_mapping")
            # A few windows at a time: a long judgment is dozens of windows, and
            # one pass over all of them can exhaust memory.
            chunks = []
            windows = encoding["input_ids"].shape[0]
            for first in range(0, windows, PREDICTION_WINDOW_BATCH):
                inputs = {
                    key: value[first:first + PREDICTION_WINDOW_BATCH].to(device)
                    for key, value in encoding.items()
                }
                with torch.no_grad():
                    chunks.append(clf(**inputs).logits.cpu())
            logits = torch.cat(chunks)

            # One flat token stream over all the windows: merge_word_spans
            # reads each word from the first window that saw it, so a word two
            # windows both covered is counted once.
            word_ids, token_offsets, labels, scores = [], [], [], []
            for window in range(len(encoding["input_ids"])):
                predicted = logits[window].argmax(-1).tolist()
                window_scores = torch.softmax(logits[window], dim=-1).tolist()
                for word_id, (start, end), index, token_scores in zip(
                    encoding.word_ids(batch_index=window),
                    offsets[window].tolist(),
                    predicted,
                    window_scores,
                ):
                    word_ids.append(word_id)
                    token_offsets.append((start, end))
                    labels.append(label_list[index])
                    scores.append(token_scores[index])

            entities = [
                {
                    "start": span["start"],
                    "end": span["end"],
                    "type": span["type"],
                    "text": text[span["start"]:span["end"]],
                    "score": round(sum(span["scores"]) / len(span["scores"]), 4),
                }
                for span in merge_word_spans(text, word_ids, token_offsets, labels, scores)
            ]
            types = list(dict.fromkeys(entity["type"] for entity in entities))
            predictions.append({
                "text": text,
                "prediction": ", ".join(types),
                "entities": entities,
            })

    elif model_type in SEQ2SEQ_TYPES:
        import torch

        tokenizer = model_data.get("tokenizer")
        if tokenizer is None:
            raise ValueError("Tokenizer not found in saved model")
        prefix = model_data.get("prefix") or ""
        max_source_length = int(model_data.get("max_source_length") or 512)
        max_target_length = int(model_data.get("max_target_length") or 128)

        clf.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device)

        # The raw texts, as the model was trained on them (never cleaned), a
        # few at a time: beam search over long documents is the heaviest pass.
        summaries = {}
        # A blank text has nothing to summarise: it gets an empty summary rather
        # than one the model invents from the prefix alone.
        written = [index for index, text in enumerate(texts) if text is not None and str(text).strip()]
        for first in range(0, len(written), PREDICTION_WINDOW_BATCH):
            indexes = written[first : first + PREDICTION_WINDOW_BATCH]
            batch = [str(texts[index]) for index in indexes]
            inputs = tokenizer(
                [prefix + text for text in batch],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_source_length,
            ).to(device)
            with torch.no_grad():
                outputs = clf.generate(**inputs, max_new_tokens=max_target_length, num_beams=4)
            for index, summary in zip(indexes, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
                summaries[index] = summary.strip()
        for index, text in enumerate(texts):
            predictions.append({"text": text, "prediction": summaries.get(index, "")})
    elif model_type in ("legalbert", "roberta"):
        import torch

        tokenizer = model_data.get("tokenizer")
        if tokenizer is None:
            raise ValueError("Tokenizer not found in saved model")

        clf.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device)

        for i, text in enumerate(cleaned):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = clf(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=-1).item())

            label = (
                label_encoder.inverse_transform([pred_idx])[0]
                if label_encoder
                else str(pred_idx)
            )
            classes = label_encoder.classes_ if label_encoder else list(range(len(probs)))
            predictions.append({
                "text": texts[i],
                "prediction": label,
                "confidence": round(probs[pred_idx], 4),
                "all_confidences": {
                    str(cls): round(probs[j], 4)
                    for j, cls in enumerate(classes)
                },
            })

    else:
        raise ValueError(f"Prediction not supported for model type: {model_type}")

    return {"predictions": predictions, "model_type": model_type}


def process_explanation_task(task_info, model_storage_path):
    """Runs one explanation task and returns the result dict for the caller to
    push into Redis. Does not touch the database."""
    model_id = task_info["model_id"]
    text = task_info["text"]
    method = task_info.get("method", "lime")  # 'lime' or 'shap'

    logger.info("%s explanation for model %s", method.upper(), model_id)

    model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = joblib.load(model_path)

    if method == "lime":
        explanation = generate_lime_explanation(model_data, text)
    elif method == "shap":
        explanation = generate_shap_explanation(model_data, text)
    else:
        raise ValueError(f"Unknown explanation method: {method}")

    return {
        "explanation": explanation,
        "text": text,
        "method": method,
        "model_id": model_id,
        "status": "completed",
    }
