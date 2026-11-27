from flask import Blueprint, request, jsonify, current_app, has_app_context
from ..auth import jwt_required, get_jwt_identity
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from ..db import db
from ..dataset_creation import DatasetInvalid, create_dataset
from ..models import AnnotationTask, Dataset, LLMPrompt
from ..annotation_format import label_from_assignment, label_text, stored_payload
from ..annotation_queue import (
    LABEL_COLUMNS,
    create_annotation_tasks,
    file_entity_rows,
    file_labels,
    read_rows,
    read_rows_exact,
)
from ..llm.suggest import current_prompt
from ..kit_exchange import ExchangeError, rows_from_task_data, task_data_from_dataset
from ..text_terms import (
    LABEL_COL_NAMES,
    TEXT_COL_NAMES,
    find_column,
    label_values,
    text_values,
    texts_containing,
)
import logging
import pandas as pd
import os
import uuid
import json
from .llm import SUPPORTED_TASK_TYPES

datasets_bp = Blueprint("datasets", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}
# The row cap when the host sets none. Hosts set LEGAL_ML_MAX_ROWS to a whole
# number, or to a function returning one (read on every upload, so a cap an
# admin changes applies at once).
MAX_ROWS = 1000


def max_rows():
    """The row cap for a new dataset: the host's LEGAL_ML_MAX_ROWS, else MAX_ROWS."""
    cap = current_app.config.get("LEGAL_ML_MAX_ROWS") if has_app_context() else None
    if callable(cap):
        cap = cap()
    if cap is None:
        return MAX_ROWS
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"LEGAL_ML_MAX_ROWS must be a whole number above 0, not {cap!r}")
    return cap

# Single source of truth for upload task types. The frontend renders this
# catalogue (via GET /datasets/task-types) so its task guidance cannot drift
# from what the backend actually accepts and validates.
TASK_TYPES = [
    {
        "value": "classification",
        "label": "Text Classification",
        "summary": "Give each short piece of text one label.",
        "when_to_use": (
            "Use when every row is a passage (a clause, sentence or paragraph) "
            "and you want one category for it, e.g. deontic type or sentiment."
        ),
        "label_meaning": (
            "The single category you want predicted for the text (e.g. "
            "Obligation, Permission, Prohibition)."
        ),
        "required_columns": ["text", "label"],
        "example": 'text="Member States shall adopt these laws", label="Obligation"',
    },
    {
        "value": "document_classification",
        "label": "Document Classification",
        "summary": "Give each whole document one label.",
        "when_to_use": (
            "Same mechanics as Text Classification, but each row is an entire "
            "document (a full ruling, contract or judgment) rather than a "
            "passage."
        ),
        "label_meaning": (
            "The single category for the whole document (e.g. case outcome or "
            "subject area)."
        ),
        "required_columns": ["text", "label"],
        "example": 'text="<full court ruling>", label="criminal"',
    },
    {
        "value": "ner",
        "label": "Named Entity Recognition",
        "summary": "Find and label spans of text (people, organisations, dates).",
        "when_to_use": (
            "Use when you need to locate specific spans inside the text — not "
            "one label for the whole text. This is the main difference from "
            "classification: the output marks entities, not the row."
        ),
        "label_meaning": (
            "Entity spans in the 'entities' column, as JSON with character "
            "offsets and a type (e.g. PERSON, ORG, DATE) per span."
        ),
        "required_columns": ["text", "entities"],
        "example": 'text="John Smith v. ABC Corp", entities=\'[{"text": "John Smith", "start": 0, "end": 10, "type": "PERSON"}]\'',
    },
    {
        "value": "named_entity_recognition",
        "label": "Named Entity Recognition (Alt)",
        "summary": "Alias of Named Entity Recognition for imported datasets.",
        "when_to_use": (
            "Identical to Named Entity Recognition; choose it only if your "
            "existing data uses this task name."
        ),
        "label_meaning": "Same as Named Entity Recognition.",
        "required_columns": ["text", "entities"],
        "example": 'text="Legal case text", entities="PERSON, ORG, DATE"',
    },
    {
        "value": "summarization",
        "label": "Text Summarization",
        "summary": "Produce a short summary from a longer text.",
        "when_to_use": (
            "Use when you have long documents and the matching short summary "
            "you want the model to learn to produce."
        ),
        "label_meaning": "The target summary text, stored in the 'summary' column.",
        "required_columns": ["text", "summary"],
        "example": 'text="<long legal document>", summary="<brief summary>"',
    },
    {
        "value": "topic_modeling",
        "label": "Topic Modeling",
        "summary": "Discover recurring themes; no labels needed.",
        "when_to_use": (
            "Unsupervised: use when you have text but no labels and want to "
            "discover themes. You can name the topics after training."
        ),
        "label_meaning": "None — unsupervised, so there is no label column.",
        "required_columns": ["text"],
        "example": 'text="<legal document content>"',
    },
    {
        "value": "clustering",
        "label": "Document Clustering",
        "summary": "Group similar documents; no labels needed.",
        "when_to_use": (
            "Unsupervised: use when you want documents grouped by similarity "
            "rather than by theme keywords. You can name the clusters after "
            "training."
        ),
        "label_meaning": "None — unsupervised, so there is no label column.",
        "required_columns": ["text"],
        "example": 'text="<document content>"',
    },
    {
        "value": "judgment_prediction",
        "label": "Judgment Prediction",
        "summary": "Predict a case outcome from the facts.",
        "when_to_use": (
            "Use when each row is a case's facts and you want to predict the "
            "outcome."
        ),
        "label_meaning": (
            "The case outcome to predict (e.g. guilty / not_guilty), stored in "
            "the 'judgment' column."
        ),
        "required_columns": ["text", "judgment"],
        "example": 'text="<case facts>", judgment="guilty"',
    },
]

VALID_TASK_TYPES = [t["value"] for t in TASK_TYPES]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_dataset_file(file_path, task_type, annotation_mode=False):
    """Validate dataset file and return basic info

    Args:
        file_path: Path to the dataset file
        task_type: Type of task (classification, ner, etc.)
        annotation_mode: If True, label columns are optional. If False, they are required.
    """
    # Outside the try: a misconfigured cap is the host's error, not the file's.
    cap = max_rows()
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            return None, "Unsupported file format"

        # Check row count limit
        if len(df) > cap:
            return None, f"Dataset exceeds maximum allowed rows ({cap})"

        # Enhanced validation based on task type
        columns = [col.lower() for col in df.columns]

        if task_type in ["classification", "document_classification"]:
            # Must have text column
            if "text" not in columns:
                return None, "Classification datasets must have a 'text' column"
            # Label column is required only for training
            if not annotation_mode and "label" not in columns:
                return (
                    None,
                    "Classification datasets for training must have a 'label' column",
                )

        elif task_type in ["ner", "named_entity_recognition"]:
            # Must have text column
            if "text" not in columns:
                return None, "NER datasets must have a 'text' column"
            # Entities column is required only for training
            if not annotation_mode and "entities" not in columns:
                return None, "NER datasets for training must have an 'entities' column"

        elif task_type in ["summarization"]:
            # Should have text column
            if "text" not in columns:
                return None, "Summarization datasets must have a 'text' column"
            # Summary column required only for training
            if not annotation_mode and "summary" not in columns:
                return (
                    None,
                    "Summarization datasets for training must have a 'summary' column",
                )

        elif task_type in ["topic_modeling", "clustering"]:
            # Should have text column (no label needed for these)
            if "text" not in columns:
                return (
                    None,
                    "Topic modeling/clustering datasets must have a 'text' column",
                )

        elif task_type in ["judgment_prediction"]:
            # Should have text column
            if "text" not in columns:
                return None, "Judgment prediction datasets must have a 'text' column"
            # Judgment column required only for training
            if not annotation_mode and "judgment" not in columns:
                return (
                    None,
                    "Judgment prediction datasets for training must have a 'judgment' column",
                )

        return df, None

    except Exception as e:
        return None, f"Error reading file: {str(e)}"


@datasets_bp.route("/", methods=["GET"])
@jwt_required()
def get_datasets():
    try:
        owner_id = get_jwt_identity()
        # Get all datasets for the user that are not already associated with a project
        # (optional: can include all datasets by removing the project_id filter)
        datasets = (
            Dataset.query.filter_by(owner_id=owner_id)
            .order_by(Dataset.created_at.desc())
            .all()
        )

        return (
            jsonify(
                {
                    "success": True,
                    "datasets": [dataset.to_dict() for dataset in datasets],
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@datasets_bp.route("/<dataset_id>", methods=["GET"])
@jwt_required()
def get_dataset(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify({"dataset": dataset.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _too_large_message():
    limit = current_app.config.get("MAX_CONTENT_LENGTH")
    if not limit:
        return "That file is too large."
    return f"That file is too large. Upload a file under {limit / (1024 * 1024):g} MB."


@datasets_bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_dataset():
    try:
        owner_id = get_jwt_identity()

        # Check if file is present
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": "File type not allowed. Supported formats: CSV, XLSX, JSON"
                    }
                ),
                400,
            )

        # Get form data
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        task_type = request.form.get("task_type", "").strip()
        annotation_mode = request.form.get("annotation_mode", "false").lower() == "true"
        labels_str = request.form.get("labels", "")

        if not name:
            return jsonify({"error": "Dataset name is required"}), 400

        if not task_type:
            return jsonify({"error": "Task type is required"}), 400

        file_path = os.path.join(
            current_app.config["UPLOAD_FOLDER"],
            f"{uuid.uuid4()}_{secure_filename(file.filename)}",
        )
        file.save(file_path)

        try:
            dataset = create_dataset(
                owner_id=owner_id,
                file_path=file_path,
                name=name,
                description=description,
                task_type=task_type,
                annotation_mode=annotation_mode,
                labels=_parse_labels(labels_str),
            )
        except DatasetInvalid as refused:
            os.remove(file_path)
            return jsonify({"error": str(refused)}), 400
        except Exception:
            # Anything else is the host's to handle (BAIGLE's error handler
            # logs it and shows a generic message); keep no orphaned file.
            db.session.rollback()
            os.remove(file_path)
            raise

        return (
            jsonify({"message": "Dataset uploaded successfully", "dataset": dataset.to_dict()}),
            201,
        )

    except RequestEntityTooLarge:
        # Raised while the form is read, before anything is saved (BAIGLE #99).
        return jsonify({"error": _too_large_message()}), 413


KIT_FORMAT = "legal-annotation-kit"


@datasets_bp.route("/<dataset_id>/annotations/export", methods=["GET"])
@jwt_required()
def export_annotations(dataset_id):
    """The Dataset's annotation tasks as vue-legal-annotation-kit TaskData."""
    dataset = Dataset.query.filter_by(id=dataset_id, owner_id=get_jwt_identity()).first()
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    if request.args.get("format", KIT_FORMAT) != KIT_FORMAT:
        return jsonify({"error": f"Annotations export as {KIT_FORMAT} only."}), 400

    tasks = AnnotationTask.query.filter_by(dataset_id=dataset.id).all()
    response = jsonify(task_data_from_dataset(dataset, tasks))
    filename = secure_filename(f"{dataset.name}-annotations.json") or "annotations.json"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response, 200


@datasets_bp.route(f"/import/{KIT_FORMAT}", methods=["POST"])
@jwt_required()
def import_kit_task():
    """A vue-legal-annotation-kit task file as a new Dataset. Each document is an
    item; the Assignment kept for it is saved on its task as a human label, so
    the Dataset can train at once and the kit can reopen every item as it was."""
    try:
        owner_id = get_jwt_identity()
        file = request.files.get("file")
        name = request.form.get("name", "").strip()
        if not file or not file.filename:
            return jsonify({"error": "No file provided"}), 400
        if not name:
            return jsonify({"error": "Dataset name is required"}), 400
        try:
            plan = rows_from_task_data(json.load(file.stream))
        except ValueError as e:  # ExchangeError, or a file that is not JSON
            message = str(e) if isinstance(e, ExchangeError) else "The file is not JSON."
            return jsonify({"error": message}), 400

        label_name = LABEL_COLUMNS[plan.task_type]
        # CSV, as training reads the stored file with pd.read_csv. Labels stay blank
        # in the file: they live on the tasks, with their Assignments.
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_kit_import.csv")
        pd.DataFrame({"text": [text for text, _ in plan.items], label_name: ""}).to_csv(file_path, index=False)
        try:
            dataset = create_dataset(
                owner_id=owner_id,
                file_path=file_path,
                name=name,
                description=request.form.get("description", "").strip(),
                task_type=plan.task_type,
                annotation_mode=True,
                labels=plan.labels,
            )
            create_annotation_tasks(dataset)
            db.session.flush()
            tasks = {t.item_index: t for t in AnnotationTask.query.filter_by(dataset_id=dataset.id)}
            labelled = 0
            for index, (text, assignment) in enumerate(plan.items):
                if assignment is None:
                    continue
                task = tasks[index]
                task.annotation_label = label_from_assignment(plan.task_type, assignment, text, plan.labels)
                task.annotation_payload = stored_payload(assignment, task)
                task.status = "annotated"
                task.label_source = "human"
                labelled += 1
            db.session.commit()
        except DatasetInvalid as refused:
            db.session.rollback()
            os.remove(file_path)
            return jsonify({"error": str(refused)}), 400
        except Exception:
            # create_dataset has already committed the Dataset; leave nothing half made.
            db.session.rollback()
            Dataset.query.filter_by(owner_id=owner_id, file_path=file_path).delete()
            db.session.commit()
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

        return jsonify({"dataset": dataset.to_dict(), "labelled": labelled, "set_aside": plan.set_aside}), 201

    except RequestEntityTooLarge:
        return jsonify({"error": _too_large_message()}), 413


def _parse_labels(labels_str):
    """Labels from the upload form: a JSON array, or comma-separated values."""
    if labels_str.startswith("["):
        try:
            return json.loads(labels_str)
        except ValueError:
            pass
    return [label.strip() for label in labels_str.split(",") if label.strip()]


# A label list a person types: kept small so a request stays small.
MAX_LABELS = 50
MAX_LABEL_LENGTH = 100


class _LabelSetRefused(Exception):
    pass


def _label_set_from(body):
    """The names in a `{"labels": [...]}` body, trimmed and checked."""
    if not isinstance(body, dict) or set(body) != {"labels"}:
        raise _LabelSetRefused("Send labels as a list of names.")
    names = body["labels"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise _LabelSetRefused("Send labels as a list of names.")
    if len(names) > MAX_LABELS:
        raise _LabelSetRefused(f"Use at most {MAX_LABELS} labels.")
    names = [name.strip() for name in names]
    if any(not name for name in names):
        raise _LabelSetRefused("Label names can't be empty.")
    if any(len(name) > MAX_LABEL_LENGTH for name in names):
        raise _LabelSetRefused(f"Keep each label name to {MAX_LABEL_LENGTH} characters or fewer.")
    if len({name.casefold() for name in names}) < len(names):
        raise _LabelSetRefused("Two labels have the same name.")
    if len(names) < 2:
        raise _LabelSetRefused("Use at least two labels.")
    return names


def _used_labels(dataset_id):
    """Every label a labelled item of the Dataset has, from any source."""
    rows = (
        db.session.query(AnnotationTask.annotation_label)
        .filter(
            AnnotationTask.dataset_id == dataset_id,
            AnnotationTask.status == "annotated",
            AnnotationTask.annotation_label.isnot(None),
        )
        .distinct()
        .all()
    )
    return {label for (label,) in rows}


@datasets_bp.route("/<dataset_id>/label-set", methods=["PUT"])
@jwt_required()
def set_label_set(dataset_id):
    """Name a single-label Dataset's labels, or change them. Free until an item
    is labelled; after that the list keeps every label in use and may only add
    to it or change unused ones. The LLM prompt and active-learning state are
    left alone: both read `dataset.labels` when they run."""
    dataset = Dataset.query.filter_by(id=dataset_id, owner_id=get_jwt_identity()).first()
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    if dataset.task_type not in SUPPORTED_TASK_TYPES:
        return (
            jsonify({"error": "Labels can only be set for single-label tasks such as Text Classification."}),
            400,
        )
    try:
        names = _label_set_from(request.get_json(silent=True))
    except _LabelSetRefused as refusal:
        return jsonify({"error": str(refusal)}), 400
    missing = sorted(_used_labels(dataset.id) - set(names))
    if missing:
        return (
            jsonify({"error": f"'{missing[0]}' is already used on labelled items, so it can't be renamed or removed."}),
            400,
        )
    dataset.labels = names
    db.session.commit()
    return jsonify({"dataset": dataset.to_dict()}), 200


PRACTICE_SUFFIX = " (practice)"


def _practice_name(name):
    """``"<name> (practice)"``, cutting ``name`` so the whole fits the column."""
    room = Dataset.name.type.length - len(PRACTICE_SUFFIX)
    return name[:room].rstrip() + PRACTICE_SUFFIX


def _practice_labels(dataset, df):
    """The copy's label list: the original's, or, when it names fewer than two,
    the labels its items have (from the file or a person), in first-seen order."""
    if isinstance(dataset.labels, list) and len(dataset.labels) >= 2:
        return list(dataset.labels)
    # As the tasks store them: a 0/1 column read as floats gives "1", not "1.0".
    seen = [text for text in map(label_text, file_labels(df, dataset).tolist()) if text]
    seen += sorted(_used_labels(dataset.id))
    return list(dict.fromkeys(seen)) or dataset.labels


@datasets_bp.route("/<dataset_id>/practice-copy", methods=["POST"])
@jwt_required()
def make_practice_copy(dataset_id):
    """An unlabelled copy of a labelled single-label Dataset, to practise labelling
    on without touching the original's labels. The copy has the same text, owner,
    project and label list, the original's current prompt as its first prompt
    version, and no labels anywhere: not in its file, not on its tasks."""
    owner_id = get_jwt_identity()
    original = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()
    if not original:
        return jsonify({"error": "Dataset not found"}), 404
    if original.task_type not in SUPPORTED_TASK_TYPES:
        return (
            jsonify({"error": "Practice copies can only be made of single-label tasks such as Text Classification."}),
            400,
        )
    try:
        df = read_rows(original.file_path)
    except Exception:
        return jsonify({"error": "This dataset's file can't be read, so it can't be copied."}), 400
    text_column = find_column(df, [(original.text_column or "text").lower()])
    if text_column is None:
        return jsonify({"error": "This dataset has no text column to copy."}), 400
    if file_labels(df, original).empty and not _used_labels(original.id):
        return jsonify({"error": "This dataset has no labels yet. Label it directly instead."}), 400

    # The text column and a blank label column only: any other column could
    # carry the labels under another name.
    label_name = LABEL_COLUMNS[original.task_type]
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_practice.csv")
    copy_id = None
    try:
        pd.DataFrame({"text": df[text_column].to_numpy(), label_name: ""}).to_csv(file_path, index=False)
        copy = create_dataset(
            owner_id=owner_id,
            file_path=file_path,
            name=_practice_name(original.name),
            description=original.description or "",
            task_type=original.task_type,
            annotation_mode=True,
            labels=_practice_labels(original, df),
        )
        copy_id = copy.id
        copy.project_id = original.project_id
        copy.origin_dataset_id = str(original.id)
        version, template, definitions = current_prompt(original)
        if version > 0:
            db.session.add(
                LLMPrompt(dataset_id=copy.id, version=1, template=template, definitions=dict(definitions))
            )
        create_annotation_tasks(copy)
        db.session.commit()
    except DatasetInvalid as refused:
        _discard_practice_copy(copy_id, file_path)
        return jsonify({"error": str(refused)}), 400
    except Exception:
        # create_dataset may have committed the copy already; leave nothing half made.
        _discard_practice_copy(copy_id, file_path)
        raise
    return jsonify({"dataset": copy.to_dict()}), 201


def _discard_practice_copy(copy_id, file_path):
    """Remove the file and Dataset a failed practice copy left behind.

    Best effort, and it never raises: the caller re-raises the error that
    matters. The file goes first, so a dead database connection cannot leave
    it behind. The Dataset goes by id through the ORM, so its tasks go with it,
    and only when create_dataset made one (as BAIGLE's sample project cleanup).
    """
    name = os.path.basename(file_path)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as exc:
        logger.warning(
            "practice_copy_cleanup_file_failed file=%s exception_type=%s", name, type(exc).__name__
        )
    try:
        db.session.rollback()
    except Exception:  # noqa: BLE001 - cleanup must not mask the real error
        pass
    if copy_id is None:
        return
    try:
        orphan = db.session.get(Dataset, copy_id)
        if orphan is not None:
            db.session.delete(orphan)
            db.session.commit()
    except Exception as exc:  # noqa: BLE001 - cleanup must not mask the real error
        try:
            db.session.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning(
            "practice_copy_cleanup_dataset_failed file=%s exception_type=%s", name, type(exc).__name__
        )


@datasets_bp.route("/task-types", methods=["GET"])
def get_task_types():
    """Return the upload task-type catalogue (single source of truth).

    Unauthenticated on purpose: it is static guidance used to render the upload
    form, so it can be fetched before the user is signed in. The frontend must
    not hard-code this list.
    """
    return jsonify({"task_types": TASK_TYPES}), 200


@datasets_bp.route("/limits", methods=["GET"])
def get_limits():
    """The limits an upload meets, read on each call so an admin's change shows
    at once. Unauthenticated, like the task types: upload copy shows it."""
    return jsonify({"max_rows": max_rows()}), 200


@datasets_bp.route("/<dataset_id>/preview", methods=["GET"])
@jwt_required()
def preview_dataset(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Read and return preview of dataset
        if dataset.file_path.endswith(".csv"):
            df = pd.read_csv(dataset.file_path)
        elif dataset.file_path.endswith(".xlsx"):
            df = pd.read_excel(dataset.file_path)
        elif dataset.file_path.endswith(".json"):
            df = pd.read_json(dataset.file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Return first 10 rows
        preview_data = df.head(10).to_dict("records")

        return (
            jsonify(
                {
                    "preview": preview_data,
                    "total_rows": len(df),
                    "columns": list(df.columns),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/<dataset_id>/rows", methods=["GET"])
@jwt_required()
def paginated_dataset_rows(dataset_id):
    """Paginated full-data browsing (HuggingFace-style dataset viewer).

    `preview_dataset` only returns the first 10 rows; this serves any window of
    the file so the UI can navigate the whole dataset without loading it all.
    """
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        offset = max(0, request.args.get("offset", default=0, type=int))
        limit = request.args.get("limit", default=50, type=int)
        limit = min(max(1, limit), 200)

        if dataset.file_path.endswith(".csv"):
            df = pd.read_csv(dataset.file_path)
        elif dataset.file_path.endswith(".xlsx"):
            df = pd.read_excel(dataset.file_path)
        elif dataset.file_path.endswith(".json"):
            df = pd.read_json(dataset.file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Optional filters for opening a distinctive term from EDA. They match
        # the way EDA counts texts, so the filtered total equals `in_class`.
        term = request.args.get("term", default="", type=str).strip()
        label = request.args.get("label", default=None, type=str)
        if term:
            text_col = find_column(df, TEXT_COL_NAMES)
            if text_col is None:
                return jsonify({"error": "This Dataset has no text column to search"}), 400
            df = df[texts_containing(text_values(df, text_col), term).to_numpy()]
        if label is not None:
            label_col = find_column(df, LABEL_COL_NAMES)
            if label_col is None:
                return jsonify({"error": "This Dataset has no label column to filter by"}), 400
            df = df[(label_values(df, label_col) == label).to_numpy()]

        # Optional filters for opening an entity from NER EDA: the rows whose
        # entities cell holds a span of that exact text and/or type. The cells
        # are read as EDA reads them (``file_entity_rows`` over the file as
        # written), so the filtered total equals the entity's ``rows`` there.
        entity = request.args.get("entity", default=None, type=str)
        entity_type = request.args.get("entity_type", default=None, type=str)
        if entity is not None or entity_type is not None:
            entity_col, entity_rows = file_entity_rows(read_rows_exact(dataset.file_path))
            if entity_col is None:
                return jsonify({"error": "This Dataset has no entities column to filter by"}), 400
            keep = [
                any(
                    (entity is None or span["text"] == entity)
                    and (entity_type is None or span["type"] == entity_type)
                    for span in spans or []
                )
                for spans, _ in entity_rows
            ]
            # ``keep`` is by position in the whole file, which is still each
            # row's index label after the earlier filters.
            df = df[pd.Series(keep).reindex(df.index, fill_value=False).to_numpy()]

        total = len(df)
        window = df.iloc[offset : offset + limit]
        # NaN is not valid JSON; pandas' `to_dict` keeps it as float('nan').
        rows = json.loads(window.to_json(orient="records", date_format="iso"))

        return (
            jsonify(
                {
                    "rows": rows,
                    "columns": list(df.columns),
                    "total_rows": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": offset + limit < total,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/<dataset_id>/preprocessing-options", methods=["GET"])
@jwt_required()
def get_preprocessing_options(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Return available preprocessing options based on task type
        preprocessing_options = {
            "text_cleaning": {
                "remove_html": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove HTML tags",
                },
                "remove_urls": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove URLs",
                },
                "remove_emails": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove email addresses",
                },
                "remove_numbers": {
                    "type": "boolean",
                    "default": False,
                    "description": "Remove numbers",
                },
                "remove_special_chars": {
                    "type": "boolean",
                    "default": False,
                    "description": "Remove special characters",
                },
                "lowercase": {
                    "type": "boolean",
                    "default": True,
                    "description": "Convert to lowercase",
                },
            },
            "tokenization": {
                "method": {
                    "type": "select",
                    "options": ["word", "sentence", "subword"],
                    "default": "word",
                    "description": "Tokenization method",
                },
                "max_length": {
                    "type": "number",
                    "min": 50,
                    "max": 512,
                    "default": 256,
                    "description": "Maximum sequence length",
                },
            },
            "feature_extraction": {
                "method": {
                    "type": "select",
                    "options": ["tfidf", "count", "hash"],
                    "default": "tfidf",
                    "description": "Feature extraction method",
                },
                "max_features": {
                    "type": "number",
                    "min": 100,
                    "max": 10000,
                    "default": 5000,
                    "description": "Maximum number of features",
                },
                "ngram_range": {
                    "type": "range",
                    "min": 1,
                    "max": 3,
                    "default": [1, 2],
                    "description": "N-gram range",
                },
            },
        }

        # Add task-specific options
        if dataset.task_type in ["ner", "named_entity_recognition"]:
            preprocessing_options["ner_specific"] = {
                "entity_types": {
                    "type": "multiselect",
                    "options": ["PERSON", "ORG", "GPE", "LAW", "CASE"],
                    "default": ["PERSON", "ORG", "GPE"],
                    "description": "Entity types to extract",
                },
                "use_spacy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use spaCy for NER preprocessing",
                },
            }

        return jsonify({"preprocessing_options": preprocessing_options}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/<dataset_id>", methods=["DELETE"])
@jwt_required()
def delete_dataset(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Delete file if it exists
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)

        # Delete dataset record (cascade will handle related models)
        db.session.delete(dataset)
        db.session.commit()

        return jsonify({"message": "Dataset deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
