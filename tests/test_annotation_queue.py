"""A dataset on a project offers every row with text for annotation: blank rows
to label, and rows the file labels as labelled tasks a person can change (#147)."""
import io
import json
import logging

import pandas as pd
import pytest

from legal_ml_core import AnnotationTask, Dataset, Model, TrainingJob, db
from legal_ml_core.annotation_queue import create_annotation_tasks, has_ner_examples, ner_examples
from legal_ml_core.compute.tasks import process_training_task
from tests.conftest import TEST_USER_ID

PROJECT_ID = "proj-queue"


def _link_dataset(app, tmp_path, rows, task_type="classification", name="data.csv"):
    """Write ``rows`` to a file and link a dataset on it to the test project."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    app.config.setdefault("_known_projects", set()).add((PROJECT_ID, TEST_USER_ID))
    dataset = Dataset(
        owner_id=TEST_USER_ID,
        name=name,
        file_path=str(path),
        row_count=len(next(iter(rows.values()))),
        columns=[{"name": column} for column in rows],
        task_type=task_type,
        project_id=PROJECT_ID,
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset.id


def _upload(client, auth_headers, csv_text):
    resp = client.post(
        "/api/datasets/upload",
        data={
            "file": (io.BytesIO(csv_text.encode()), "rows.csv"),
            "name": "Rows",
            "task_type": "classification",
        },
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    return resp.get_json()["dataset"]["id"]


def _next(client, auth_headers):
    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/queue/next", headers=auth_headers)
    assert resp.status_code == 200
    return resp.get_json()


def _tasks(client, auth_headers):
    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/tasks", headers=auth_headers)
    return resp.get_json()["tasks"]


def test_unlabelled_rows_are_queued_in_row_order(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b", "c"], "label": ["x", None, None]})

    body = _next(client, auth_headers)

    assert body["task"]["text"] == "b"
    assert body["task"]["item_index"] == 1
    assert body["stats"] == {"total": 3, "annotated": 1, "percentage": 33}


def test_whitespace_label_counts_as_blank(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": ["   ", "x"]})

    assert [(t["text"], t["status"]) for t in _tasks(client, auth_headers)] == [("a", "pending"), ("b", "annotated")]


def test_file_without_a_label_column_queues_every_row(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"]})

    assert _next(client, auth_headers)["stats"]["total"] == 2


def test_a_fully_labelled_dataset_is_all_annotated_with_its_file_labels(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": ["x", "y"]})

    body = _next(client, auth_headers)

    assert body["task"] is None
    assert body["stats"] == {"total": 2, "annotated": 2, "percentage": 100}
    assert body["message"] == "All items annotated"
    tasks = _tasks(client, auth_headers)
    assert [(t["annotation_label"], t["label_source"], t["file_label"]) for t in tasks] == [
        ("x", "file", "x"),
        ("y", "file", "y"),
    ]


def test_column_names_match_case_insensitively(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"Text": ["a", "b"], "Label": ["x", None]})

    assert [(t["text"], t["status"]) for t in _tasks(client, auth_headers)] == [("a", "annotated"), ("b", "pending")]


def test_ner_rows_are_all_queued_with_the_files_types_kept(client, auth_headers, app, tmp_path):
    # An NER file names entity types, not spans, so every row is marked by hand.
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "entities": ["PERSON, ORG", None]}, task_type="ner")

    tasks = _tasks(client, auth_headers)

    assert [(t["text"], t["status"], t["file_label"]) for t in tasks] == [("a", "pending", "PERSON, ORG"), ("b", "pending", None)]


def test_task_types_without_labels_queue_nothing(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"]}, task_type="topic_modeling")

    assert _tasks(client, auth_headers) == []


def test_summarization_is_not_queued(client, auth_headers, app, tmp_path):
    # No trainer learns from summaries, so writing them here would go nowhere.
    rows = {"text": ["a", "b"], "summary": [None, None]}
    _link_dataset(app, tmp_path, rows, task_type="summarization")

    assert _tasks(client, auth_headers) == []


def test_rows_with_blank_text_are_left_out(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", None], "label": [None, None]})

    assert [t["text"] for t in _tasks(client, auth_headers)] == ["a"]


def test_repeated_reads_do_not_duplicate_tasks(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": [None, None]})

    _next(client, auth_headers)
    _next(client, auth_headers)

    assert len(_tasks(client, auth_headers)) == 2


def test_saved_and_skipped_work_survives_later_reads(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b", "c"], "label": [None, None, None]})
    first = _next(client, auth_headers)["task"]["id"]
    client.post(
        f"/api/annotation/tasks/{first}/save", json={"annotation_label": "x"}, headers=auth_headers
    )
    second = _next(client, auth_headers)["task"]["id"]
    client.post(f"/api/annotation/tasks/{second}/skip", headers=auth_headers)

    body = _next(client, auth_headers)

    assert body["task"]["text"] == "c"
    assert body["stats"] == {"total": 3, "annotated": 1, "percentage": 33}
    assert sorted(t["status"] for t in _tasks(client, auth_headers)) == [
        "annotated",
        "pending",
        "skipped",
    ]


def test_stats_count_rows_before_the_queue_is_opened(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": [None, None]})

    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/queue/stats", headers=auth_headers)

    assert resp.get_json()["stats"]["pending"] == 2


def _rewrite(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "data.csv", index=False)


def test_a_fully_labelled_file_is_read_only_once(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": ["x", "y"]})
    _next(client, auth_headers)

    # Only a second read of the file could find this change.
    _rewrite(tmp_path, {"text": ["a", "b", "c"], "label": ["x", None, None]})

    assert _next(client, auth_headers)["stats"] == {"total": 2, "annotated": 2, "percentage": 100}


def test_an_uploaded_labelled_file_becomes_labelled_tasks(client, auth_headers, app, tmp_path):
    app.config.setdefault("_known_projects", set()).add((PROJECT_ID, TEST_USER_ID))
    dataset_id = _upload(client, auth_headers, "text,label\na,x\nb,y\n")
    dataset = db.session.get(Dataset, dataset_id)
    dataset.project_id = PROJECT_ID
    db.session.commit()

    assert _next(client, auth_headers)["stats"] == {"total": 2, "annotated": 2, "percentage": 100}


def test_upload_counts_rows_to_label_the_way_the_queue_does(client, auth_headers):
    # One blank label, one whitespace label, one blank text, one labelled.
    dataset_id = _upload(client, auth_headers, 'text,label\na,\nb,"   "\n,\nc,x\n')

    columns = db.session.get(Dataset, dataset_id).columns

    assert [c["rows_to_label"] for c in columns if c["name"] == "text"] == [2]


def test_reading_the_queue_records_the_count_for_older_datasets(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a", "b", None], "label": ["  ", "x", None]})

    _next(client, auth_headers)

    columns = db.session.get(Dataset, dataset_id).columns
    assert [c["rows_to_label"] for c in columns if c["name"] == "text"] == [1]


def _queued_task(app, tmp_path, client, auth_headers):
    _link_dataset(app, tmp_path, {"text": ["a"], "label": [None]})
    return _next(client, auth_headers)["task"]["id"]


def test_saving_an_empty_label_is_refused(client, auth_headers, app, tmp_path):
    task_id = _queued_task(app, tmp_path, client, auth_headers)

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save", json={"annotation_label": ""}, headers=auth_headers
    )

    assert resp.status_code == 400
    assert db.session.get(AnnotationTask, task_id).status == "pending"


def test_saving_a_whitespace_label_is_refused(client, auth_headers, app, tmp_path):
    task_id = _queued_task(app, tmp_path, client, auth_headers)

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save",
        json={"annotation_label": "   "},
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert db.session.get(AnnotationTask, task_id).status == "pending"


def test_a_passage_with_no_entities_can_be_saved(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a"], "entities": [None]}, task_type="ner")
    task_id = _next(client, auth_headers)["task"]["id"]

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save",
        json={"annotation_label": "[]"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert db.session.get(AnnotationTask, task_id).status == "annotated"


def test_unreadable_file_leaves_the_queue_empty(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a"], "label": [None]})
    (tmp_path / "data.csv").unlink()

    assert _next(client, auth_headers)["stats"]["total"] == 0
    assert db.session.get(Dataset, dataset_id) is not None


def test_another_owners_dataset_on_the_project_is_not_queued(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a"], "label": [None]})
    dataset = db.session.get(Dataset, dataset_id)
    dataset.owner_id = "22222222-2222-2222-2222-222222222222"
    db.session.commit()

    assert _next(client, auth_headers)["stats"]["total"] == 0
    assert AnnotationTask.query.count() == 0


def _train(app, fake_redis, dataset_id, task_type="classification"):
    dataset = db.session.get(Dataset, dataset_id)
    model = Model(
        owner_id=TEST_USER_ID,
        dataset_id=dataset_id,
        name="M",
        model_type="naive_bayes",
        parameters={},
        status="training",
    )
    db.session.add(model)
    db.session.flush()
    job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="pending")
    db.session.add(job)
    db.session.commit()
    process_training_task(
        {
            "job_id": job.id,
            "model_id": model.id,
            "dataset_id": dataset_id,
            "model_type": "naive_bayes",
            "parameters": {},
            "dataset_path": dataset.file_path,
            "task_type": task_type,
            "preprocessing_options": {},
        },
        fake_redis,
        app.config["MODEL_FOLDER"],
    )
    return db.session.get(Model, model.id), db.session.get(TrainingJob, job.id)


def _sentences(n):
    return [f"clause number {i} about the contract" for i in range(n)]


def test_saved_annotations_become_training_labels(client, auth_headers, app, fake_redis, tmp_path):
    rows = {"text": _sentences(16), "label": ["civil", "criminal"] * 6 + [None] * 4}
    dataset_id = _link_dataset(app, tmp_path, rows)
    for _ in range(3):
        task = _next(client, auth_headers)["task"]
        client.post(
            f"/api/annotation/tasks/{task['id']}/save",
            json={"annotation_label": "labour"},
            headers=auth_headers,
        )

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    # The last row stays unlabelled: it must be left out, not become a class.
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal", "labour"]


def test_a_changed_file_label_trains_with_the_change(client, auth_headers, app, fake_redis, tmp_path):
    rows = {"text": _sentences(12), "label": ["civil", "criminal"] * 6}
    dataset_id = _link_dataset(app, tmp_path, rows)
    _next(client, auth_headers)
    for task in _tasks(client, auth_headers)[:2]:
        client.post(f"/api/annotation/tasks/{task['id']}/save", json={"annotation_label": "labour"}, headers=auth_headers)

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal", "labour"]


def test_file_without_a_label_column_trains_on_saved_annotations(
    client, auth_headers, app, fake_redis, tmp_path
):
    dataset_id = _link_dataset(app, tmp_path, {"text": _sentences(10)})
    for label in ["civil", "criminal"] * 5:
        task = _next(client, auth_headers)["task"]
        client.post(
            f"/api/annotation/tasks/{task['id']}/save",
            json={"annotation_label": label},
            headers=auth_headers,
        )

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal"]


def test_training_uses_the_annotated_column_not_the_second_one(
    client, auth_headers, app, fake_redis, tmp_path
):
    rows = {"text": _sentences(10), "court": ["Hoge Raad", "Rechtbank"] * 5}
    dataset_id = _link_dataset(app, tmp_path, rows)
    for label in ["civil", "criminal"] * 5:
        task = _next(client, auth_headers)["task"]
        client.post(
            f"/api/annotation/tasks/{task['id']}/save",
            json={"annotation_label": label},
            headers=auth_headers,
        )

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal"]


def test_training_reads_the_text_column_wherever_it_sits(
    client, auth_headers, app, fake_redis, tmp_path
):
    rows = {
        "court": ["Hoge Raad"] * 10,
        "text": ["contract breach"] * 5 + ["theft and robbery"] * 5,
        "label": ["civil"] * 5 + ["criminal"] * 5,
    }
    dataset_id = _link_dataset(app, tmp_path, rows)

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal"]


def test_a_dataset_queued_before_gets_its_file_labelled_rows_once(client, auth_headers, app, tmp_path):
    """A Dataset whose tasks were made under the old rule (blank rows only)."""
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a", "b", "c"], "label": ["x", None, "y"]})
    db.session.add(AnnotationTask(dataset_id=dataset_id, item_index=1, text="b", status="annotated", annotation_label="z", label_source="human"))
    db.session.commit()

    _next(client, auth_headers)
    _next(client, auth_headers)

    tasks = _tasks(client, auth_headers)
    assert [(t["item_index"], t["annotation_label"], t["label_source"]) for t in tasks] == [
        (0, "x", "file"),
        (1, "z", "human"),
        (2, "y", "file"),
    ]


def test_changing_a_file_label_makes_it_a_human_label_and_keeps_the_files(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a"], "label": ["x"]})
    _next(client, auth_headers)
    task = _tasks(client, auth_headers)[0]

    client.post(f"/api/annotation/tasks/{task['id']}/save", json={"annotation_label": "y"}, headers=auth_headers)

    changed = _tasks(client, auth_headers)[0]
    assert (changed["annotation_label"], changed["label_source"], changed["file_label"]) == ("y", "human", "x")


def _review(client, auth_headers, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/tasks?{query}", headers=auth_headers)
    assert resp.status_code == 200
    return resp.get_json()


def test_labelled_items_list_in_row_order_by_label_a_page_at_a_time(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": list("abcdef"), "label": ["x", "y", None, "x", "x", "y"]})
    _next(client, auth_headers)

    everything = _review(client, auth_headers, status="annotated", dataset_id=dataset_id)
    page = _review(client, auth_headers, status="annotated", dataset_id=dataset_id, label="x", limit=2, offset=1)

    assert [t["text"] for t in everything["tasks"]] == ["a", "b", "d", "e", "f"]
    assert everything["total"] == 5
    assert [t["text"] for t in page["tasks"]] == ["d", "e"]
    assert page["total"] == 3


def test_held_back_items_can_be_listed_for_review(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": ["x", "y"]})
    _next(client, auth_headers)
    task = AnnotationTask.query.filter_by(dataset_id=dataset_id, item_index=0).one()
    task.is_holdout = True
    db.session.commit()

    assert [t["text"] for t in _review(client, auth_headers, status="annotated")["tasks"]] == ["b"]
    assert [t["text"] for t in _review(client, auth_headers, status="annotated", include_holdout=1)["tasks"]] == ["a", "b"]


def test_an_ner_item_opens_with_the_files_types_as_a_hint(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["Smith sued Acme."], "entities": ["PERSON, ORG"]}, task_type="ner")
    task = _next(client, auth_headers)["task"]

    opened = client.get(f"/api/annotation/tasks/{task['id']}/assignment", headers=auth_headers).get_json()

    assert opened["hint"] == "PERSON, ORG"
    assert opened["file_label"] == "PERSON, ORG"
    assert opened["label_source"] is None


SPANS = [
    {"text": "Smith", "start": 0, "end": 5, "type": "PERSON"},
    {"text": "Acme", "start": 11, "end": 15, "type": "ORG"},
]


def test_ner_rows_with_json_spans_start_annotated(client, auth_headers, app, tmp_path):
    # An imported NER row carries its spans in the entities cell: no work left.
    _link_dataset(
        app,
        tmp_path,
        {"text": ["Smith sued Acme.", "No entities here."], "entities": [json.dumps(SPANS), "[]"]},
        task_type="ner",
    )

    body = _next(client, auth_headers)

    assert body["task"] is None
    assert body["stats"] == {"total": 2, "annotated": 2, "percentage": 100}
    tasks = _tasks(client, auth_headers)
    assert [(t["status"], t["label_source"], json.loads(t["annotation_label"])) for t in tasks] == [
        ("annotated", "file", SPANS),
        ("annotated", "file", []),
    ]


def test_ner_rows_keep_the_file_text_exactly(client, auth_headers, app, tmp_path):
    # "NA", "null" and numbers are what the annotator wrote: pandas must not
    # turn them into NaN or shrink them, or the spans point into other text.
    path = tmp_path / "verbatim.csv"
    path.write_text(
        "text,entities\n"
        'NA,"[{""start"": 0, ""end"": 2, ""type"": ""PERSON""}]"\n'
        'null,"[{""start"": 0, ""end"": 4, ""type"": ""ORG""}]"\n'
        '00123,"[{""start"": 0, ""end"": 5, ""type"": ""PERSON""}]"\n'
        '1.50,"[{""start"": 0, ""end"": 4, ""type"": ""ORG""}]"\n'
    )
    app.config.setdefault("_known_projects", set()).add((PROJECT_ID, TEST_USER_ID))
    dataset = Dataset(
        owner_id=TEST_USER_ID,
        name="verbatim.csv",
        file_path=str(path),
        row_count=4,
        columns=[{"name": "text"}, {"name": "entities"}],
        task_type="ner",
        project_id=PROJECT_ID,
    )
    db.session.add(dataset)
    db.session.commit()
    dataset_id = dataset.id

    # The read path sees the file's spans and creates no tasks.
    fit = client.get(f"/api/models/available?dataset_id={dataset_id}", headers=auth_headers).get_json()
    assert fit["models"]["distilbert_ner"]["usable"] is True
    assert AnnotationTask.query.count() == 0

    _next(client, auth_headers)

    tasks = _tasks(client, auth_headers)
    assert [(t["text"], t["status"]) for t in tasks] == [
        ("NA", "annotated"),
        ("null", "annotated"),
        ("00123", "annotated"),
        ("1.50", "annotated"),
    ]
    assert [json.loads(t["annotation_label"])[0]["text"] for t in tasks] == [
        "NA",
        "null",
        "00123",
        "1.50",
    ]


def test_holdout_tasks_are_not_ner_examples(app, tmp_path):
    dataset_id = _link_dataset(
        app,
        tmp_path,
        # Both rows' spans fit their text, so both start annotated.
        {"text": ["Smith sued Acme.", "Smith sued Acme."], "entities": [json.dumps(SPANS), json.dumps(SPANS)]},
        task_type="ner",
    )
    dataset = db.session.get(Dataset, dataset_id)
    create_annotation_tasks(dataset)
    db.session.commit()
    AnnotationTask.query.filter_by(dataset_id=dataset_id, item_index=0).update({"is_holdout": True})
    db.session.commit()

    examples = ner_examples(db.session.get(Dataset, dataset_id))
    assert [spans for _, spans in examples] == [SPANS]

    # With every annotated task held out, there is nothing to learn from.
    AnnotationTask.query.filter_by(dataset_id=dataset_id).update({"is_holdout": True})
    db.session.commit()
    assert has_ner_examples(db.session.get(Dataset, dataset_id)) is False


def test_a_rejected_span_looking_cell_is_warned_about(client, auth_headers, app, tmp_path, caplog):
    # The cell is JSON but its offsets do not fit the row's text.
    _link_dataset(
        app,
        tmp_path,
        {"text": ["Smith sued Acme."], "entities": ['[{"start": 0, "end": 99, "type": "PERSON"}]']},
        task_type="ner",
    )

    with caplog.at_level(logging.WARNING, logger="legal_ml_core.annotation_queue"):
        _next(client, auth_headers)

    warning = next(record for record in caplog.records if "looks like spans" in record.message)
    assert "row 0" in warning.getMessage()
    assert "Smith" not in warning.getMessage()


@pytest.mark.parametrize(
    "entities",
    [
        "PERSON, ORG",
        "not json",
        '[{"start": 0, "end": 99, "type": "PERSON"}]',
        '[{"start": 0, "end": 5, "type": "PERSON", "text": "Wrong"}]',
        '[{"start": 0, "end": 5, "type": ""}]',
        '[{"start": 0, "end": 5, "type": "PERSON"}, {"start": 3, "end": 8, "type": "ORG"}]',
        '[{"start": 11, "end": 15, "type": "ORG"}, {"start": 0, "end": 5, "type": "PERSON"}]',
        '[{"start": 0, "end": 5, "label": "PERSON"}]',
    ],
)
def test_ner_entities_that_are_not_spans_stay_pending(client, auth_headers, app, tmp_path, entities):
    _link_dataset(app, tmp_path, {"text": ["Smith sued Acme."], "entities": [entities]}, task_type="ner")

    tasks = _tasks(client, auth_headers)

    assert [(t["status"], t["annotation_label"], t["file_label"]) for t in tasks] == [
        ("pending", None, entities)
    ]
