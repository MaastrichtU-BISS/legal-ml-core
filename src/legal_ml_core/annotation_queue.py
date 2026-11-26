"""Dataset rows becoming annotation work, and saved work becoming labels."""
import json
import logging

import pandas as pd
from sqlalchemy.exc import IntegrityError

from .annotation_format import SPAN_TYPES, _stored_entities, label_text
from .db import db
from .models import AnnotationTask, Dataset

logger = logging.getLogger(__name__)

# The column holding each task type's answer, as required at upload. Task types
# left out (topic modeling, clustering) have no labels, so no annotation work.
LABEL_COLUMNS = {
    "classification": "label",
    "document_classification": "label",
    "ner": "entities",
    "named_entity_recognition": "entities",
    "summarization": "summary",
    "judgment_prediction": "judgment",
}

# Task types whose trainer learns from the label column. NER trains from spans
# held in ``annotation_label`` instead, through ``ner_examples`` below, and stays
# out of this set so the classification paths and active learning do not see it.
TRAINED_ON_LABELS = {"classification", "document_classification", "judgment_prediction"}

# Task types offered on the annotation screen. Summarization is left out: its
# trainer reads the summary column straight from the file, and the annotation
# screen has no place to write a summary.
QUEUED_TASK_TYPES = TRAINED_ON_LABELS | {"ner", "named_entity_recognition"}

# Label sources an LLM had a hand in. Holdout metrics rest on human labels only.
LLM_LABEL_SOURCES = ("llm_confirmed", "llm_bulk")

# The trainer's stratified split needs two of a class: one a side.
MIN_TO_SPLIT = 2

# Key on the text column's entry in ``Dataset.columns``: how many rows of the file
# need a label. The host reads it too, so both count what the queue would offer.
ROWS_TO_LABEL = "rows_to_label"

# Key on the same entry: every row with text has a task, file-labelled ones
# included, so the file need not be read again for the queue (#147).
ALL_ROWS_QUEUED = "all_rows_queued"

# Where a task's label came from when the uploaded file gave it.
FILE_LABEL_SOURCE = "file"


def _find_column(df, name):
    """The column of ``df`` called ``name``, ignoring case, else ``None``."""
    for column in df.columns:
        if str(column).lower() == name.lower():
            return column
    return None


def _is_blank(values):
    return values.isna() | values.astype(str).str.strip().eq("")


def read_rows(file_path):
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    if file_path.endswith(".json"):
        return pd.read_json(file_path)
    return pd.read_csv(file_path)


def read_rows_exact(file_path):
    """``read_rows`` with pandas' type guesses switched off, for NER files: the
    spans' offsets point into the text exactly as written, so ``NA`` and
    ``null`` stay text instead of becoming NaN, ``00123`` stays ``00123``, and a
    blank cell stays blank."""
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, dtype=str, keep_default_na=False)
    if file_path.endswith(".json"):
        return pd.read_json(file_path, dtype=False)
    return pd.read_csv(file_path, dtype=str, keep_default_na=False)


def _rows_needing_label(df, dataset):
    """Which rows need a label: those with text and a blank label.

    ``None`` when the dataset has no labels to give, or no text to read.
    """
    label_name = dataset.label_column or LABEL_COLUMNS.get(dataset.task_type)
    text_column = _find_column(df, dataset.text_column or "text")
    if not label_name or text_column is None:
        return None

    label_column = _find_column(df, label_name)
    needs_label = ~_is_blank(df[text_column])
    if label_column is not None:
        needs_label &= _is_blank(df[label_column])
    return needs_label


def file_labels(df, dataset):
    """The labels ``df`` already holds, as a series keyed by row position."""
    label_name = dataset.label_column or LABEL_COLUMNS.get(dataset.task_type)
    label_column = _find_column(df, label_name) if label_name else None
    if label_column is None:
        return pd.Series(dtype=object)
    return df.loc[~_is_blank(df[label_column]), label_column]


def rows_with_text(df, dataset):
    """Row positions of ``df`` that have text."""
    text_column = _find_column(df, dataset.text_column or "text")
    if text_column is None:
        return []
    return (~_is_blank(df[text_column])).to_numpy().nonzero()[0].tolist()


def _holdout_of(dataset):
    config = dataset.al_config
    return (config.holdout_indices or []) if config else []


def holdout_examples(df, dataset):
    """``(texts, labels)`` of the held-out rows with a human label or one from the
    file, for grading a model."""
    holdout = _holdout_of(dataset)
    text_column = _find_column(df, dataset.text_column or "text")
    labels = file_labels(df, dataset).astype(object).to_dict()
    texts_by_row = df[text_column].astype(str)
    for task in dataset.annotation_tasks:
        if task.is_holdout and task.status == "annotated" and task.label_source not in LLM_LABEL_SOURCES:
            labels[task.item_index] = task.annotation_label
    rows = [row for row in holdout if row in labels]
    # Read as the trainer reads them, so a file's 1.0 grades against the class "1".
    return [texts_by_row.iloc[row] for row in rows], [label_text(labels[row]) for row in rows]


def record_rows_to_label(dataset, df):
    """Note in ``dataset.columns`` how many rows of ``df`` need a label; caller commits."""
    needs_label = _rows_needing_label(df, dataset)
    if needs_label is None:
        return

    text_name = (dataset.text_column or "text").lower()
    dataset.columns = [
        {**column, ROWS_TO_LABEL: int(needs_label.sum())}
        if str(column["name"]).lower() == text_name
        else column
        for column in dataset.columns or []
    ]


def _all_rows_queued(dataset):
    return any(column.get(ALL_ROWS_QUEUED) for column in dataset.columns or [])


def _mark_all_rows_queued(dataset):
    text_name = (dataset.text_column or "text").lower()
    dataset.columns = [
        {**column, ALL_ROWS_QUEUED: True} if str(column["name"]).lower() == text_name else column
        for column in dataset.columns or []
    ]


def _file_spans(value, text):
    """The spans an NER file's ``entities`` cell holds, in the stored format, or
    ``None`` when it is not a span list for ``text``.

    A JSON list of span dicts (``text``, ``start``, ``end``, ``type``) with
    offsets inside ``text`` is a labelled row; anything else, a type list such as
    ``"PERSON, ORG"`` included, is not.
    """
    if isinstance(value, (list, tuple)):
        # A JSON file's cell is already a list, not JSON text.
        spans = list(value)
    elif value is None or not str(value).strip():
        return None
    else:
        try:
            spans = json.loads(str(value))
        except ValueError:
            return None
    if not isinstance(spans, list):
        return None

    checked = []
    for span in spans:
        if not isinstance(span, dict):
            return None
        start, end, kind = span.get("start"), span.get("end"), span.get("type")
        if isinstance(start, bool) or isinstance(end, bool):
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if not 0 <= start < end <= len(text):
            return None
        if not isinstance(kind, str) or not kind.strip():
            return None
        if "text" in span and span["text"] != text[start:end]:
            return None
        checked.append({"text": text[start:end], "start": start, "end": end, "type": kind})

    # In text order, and no two spans sharing a character.
    for previous, current in zip(checked, checked[1:]):
        if current["start"] < previous["end"]:
            return None
    return checked


def _checked_spans(value, text, dataset, position):
    """The spans one ``entities`` cell holds (``_file_spans``), with a warning
    when a cell that looks like spans does not fit the row's text. The warning
    names the dataset and the row, never the text."""
    spans = _file_spans(value, text)
    looks_like_spans = isinstance(value, (list, tuple)) or str(value).strip().startswith("[")
    if spans is None and looks_like_spans:
        logger.warning(
            "dataset %s row %s: entities cell looks like spans but does not fit the text",
            dataset.id,
            position,
        )
    return spans


def ner_examples(dataset):
    """``[(text, spans)]`` for an NER Dataset's annotated, non-holdout tasks.

    The raw task text is kept: cleaning it would move every span's offsets.
    """
    tasks = (
        AnnotationTask.query.filter_by(dataset_id=dataset.id, status="annotated", is_holdout=False)
        .order_by(AnnotationTask.item_index)
        .all()
    )
    return [(task.text, _stored_entities(task.annotation_label)) for task in tasks]


def ensure_dataset_tasks(dataset):
    """Create ``dataset``'s missing tasks, as reading the queue would. Write
    paths only — making a Model, starting a run, the worker: this commits, and
    on a read path (or inside the locked part of a start) that commit would
    write where nothing should, and release the lock a double-clicked Train
    serialises on.

    An imported NER Dataset's spans live in its file until its rows become
    tasks; ``has_ner_examples`` reads the file instead of creating them, so
    asking what a Dataset can train writes nothing."""
    # Only NER needs this: other task types' file labels train from the file
    # itself, and their tasks are made when a project's queue is read.
    if dataset is None or dataset.task_type not in SPAN_TYPES or _all_rows_queued(dataset):
        return
    create_annotation_tasks(dataset)
    try:
        db.session.commit()
    except IntegrityError:
        # A concurrent read created them first (uq_dataset_item).
        db.session.rollback()


def has_ner_examples(dataset):
    """Whether an NER Dataset has an entity to learn from: an annotated,
    non-holdout task whose spans are not empty (rows annotated ``[]`` teach
    nothing), or — while its rows are not all queued yet — a row of its file
    that holds spans. Read-only: the fit check runs on read paths and inside a
    start's locked section, where creating tasks would commit."""
    annotated = (
        AnnotationTask.query.filter_by(dataset_id=dataset.id, status="annotated", is_holdout=False)
        .order_by(AnnotationTask.item_index)
        .all()
    )
    if any(_stored_entities(task.annotation_label) for task in annotated):
        return True
    if _all_rows_queued(dataset):
        return False
    return _file_has_spans(dataset)


def _file_has_spans(dataset):
    """Whether any non-holdout row of ``dataset``'s file holds spans: an
    imported Dataset's rows may not have become tasks yet."""
    try:
        df = read_rows_exact(dataset.file_path)
    except Exception as e:
        logger.warning("dataset %s: cannot read rows to look for spans: %s", dataset.id, e)
        return False
    text_column = _find_column(df, dataset.text_column or "text")
    label_name = dataset.label_column or LABEL_COLUMNS.get(dataset.task_type)
    label_column = _find_column(df, label_name) if label_name else None
    if text_column is None or label_column is None:
        return False
    held_out = set(_holdout_of(dataset))
    for position in rows_with_text(df, dataset):
        if position in held_out:
            continue
        text = str(df[text_column].iloc[position])
        if _checked_spans(df[label_column].iloc[position], text, dataset, position):
            return True
    return False


def summarization_examples(file_path, dataset=None):
    """``[(text, summary)]`` of the rows of the file that have both, read
    verbatim (see ``read_rows_exact``): a summary is text, so ``NA``, ``null``
    and ``00123`` stay as written, and nothing is cleaned (cleaning would
    lowercase it). Columns are found by name, never by position; a row missing
    one of them is not an example, so it trains on nothing."""
    df = read_rows_exact(file_path)
    text_name = (dataset.text_column if dataset else None) or "text"
    summary_name = (dataset.label_column if dataset else None) or LABEL_COLUMNS["summarization"]
    text_column = _find_column(df, text_name)
    summary_column = _find_column(df, summary_name)
    if text_column is None or summary_column is None:
        raise ValueError("A summarization Dataset needs a text column and a summary column.")
    keep = ~_is_blank(df[text_column]) & ~_is_blank(df[summary_column])
    # A JSON file keeps numbers as numbers; the model reads text.
    return [
        (str(text), str(summary))
        for text, summary in zip(df.loc[keep, text_column], df.loc[keep, summary_column])
    ]


def has_summarization_examples(dataset):
    """Whether a summarization Dataset has a row with both a text and a summary:
    the fewest examples its trainer can start from. Read-only (the fit check
    runs on read paths and inside a start's locked section), and it reads the
    file exactly as the trainer will, so the two can never disagree."""
    try:
        return bool(summarization_examples(dataset.file_path, dataset))
    except Exception as e:
        logger.warning("dataset %s: cannot read rows to look for summaries: %s", dataset.id, e)
        return False


def create_annotation_tasks(dataset):
    """Add a task for every row of ``dataset`` with text that has none; caller commits.

    A row the file labels starts labelled (``label_source = "file"``), so it trains
    as before and a person can change it; a blank row is pending. An NER file's
    ``entities`` cell may hold spans, in which case the row starts labelled too;
    a cell naming entity types (``"PERSON, ORG"``) is not spans, so that row stays
    pending with the types kept as ``file_label``. A Dataset with an active
    learning loop has its new tasks held back as the loop drew them, and stops
    counting file labels on the side.
    """
    if dataset.task_type not in QUEUED_TASK_TYPES:
        return

    try:
        # An NER file is read verbatim (see read_rows_exact): its spans point
        # into the text as written, which pandas' type guessing would change.
        df = (
            read_rows_exact(dataset.file_path)
            if dataset.task_type in SPAN_TYPES
            else read_rows(dataset.file_path)
        )
    except Exception as e:
        logger.warning("dataset %s: cannot read rows to annotate: %s", dataset.id, e)
        return

    text_column = _find_column(df, dataset.text_column or "text")
    if text_column is None:
        return
    record_rows_to_label(dataset, df)

    label_name = dataset.label_column or LABEL_COLUMNS.get(dataset.task_type)
    label_column = _find_column(df, label_name) if label_name else None
    labels_rows = dataset.task_type in TRAINED_ON_LABELS
    spans_rows = dataset.task_type in SPAN_TYPES
    existing = {task.item_index for task in dataset.annotation_tasks}
    config = dataset.al_config
    held_out = set(config.holdout_indices or []) if config else set()

    added_labelled = False
    for position in rows_with_text(df, dataset):
        if position in existing:
            continue
        text = str(df[text_column].iloc[position])
        file_label = label_text(df[label_column].iloc[position]) if label_column is not None else None
        spans = _checked_spans(file_label, text, dataset, position) if spans_rows else None
        label = json.dumps(spans) if spans is not None else (file_label if labels_rows else None)
        labelled = label is not None
        added_labelled = added_labelled or labelled
        db.session.add(
            AnnotationTask(
                dataset_id=dataset.id,
                item_index=int(position),
                text=text,
                status="annotated" if labelled else "pending",
                annotation_label=label if labelled else None,
                label_source=FILE_LABEL_SOURCE if labelled else None,
                file_label=file_label,
                is_holdout=position in held_out,
            )
        )
    if config is not None and added_labelled:
        # These labels now live on tasks; counting them from the file too would
        # count them twice (a loop started before #147).
        config.file_label_counts = {}
        config.holdout_file_counts = {}
    _mark_all_rows_queued(dataset)


def ensure_project_tasks(project_id, owner_id):
    """Create the missing tasks of ``owner_id``'s datasets on ``project_id``.

    Runs when the queue is read rather than when a dataset is linked: linking is
    the host's route, in several places, and this also picks up datasets linked
    before tasks existed. Only missing tasks are added, so saved and skipped work
    is never touched; once every row has one the file is not read again. A
    Dataset queued before #147 (blank rows only) gets its file-labelled rows here.
    """
    datasets = Dataset.query.filter_by(project_id=project_id, owner_id=str(owner_id)).all()
    for dataset in datasets:
        if dataset.task_type not in QUEUED_TASK_TYPES or _all_rows_queued(dataset):
            continue
        create_annotation_tasks(dataset)
        try:
            db.session.commit()
        except IntegrityError:
            # A concurrent read created them first (uq_dataset_item).
            db.session.rollback()


def apply_saved_annotations(df, dataset):
    """``df`` as its text and label columns, in the order the trainer reads them.

    Saved annotations become labels and still-unlabelled rows are dropped, as
    are held-out rows: they grade the model, so it must never see them. In a loop,
    so is a class with one example, as the split cannot place it; the loop says so.
    """
    if dataset is None or dataset.task_type not in TRAINED_ON_LABELS:
        return df

    label_name = dataset.label_column or LABEL_COLUMNS[dataset.task_type]
    label_column = _find_column(df, label_name)
    if label_column is None:
        label_column = label_name
        df = df.assign(**{label_name: None})

    df = df.astype({label_column: object})
    for task in dataset.annotation_tasks:
        if task.status == "annotated":
            df.loc[task.item_index, label_column] = task.annotation_label

    text_column = _find_column(df, dataset.text_column or "text")
    if text_column is None:
        text_column = df.columns[0]
    keep = ~_is_blank(df[label_column]) & ~df.index.isin(_holdout_of(dataset))
    df = df.loc[keep, [text_column, label_column]]
    if dataset.al_config is not None:
        # Counted as the trainer reads them: 1, 1.0 and "1" are one class.
        labels = df[label_column].map(label_text)
        df = df[labels.map(labels.value_counts()) >= MIN_TO_SPLIT]
    return df.reset_index(drop=True)
