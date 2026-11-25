"""PUT /api/datasets/<id>/label-set names a single-label Dataset's labels."""

import pytest

from legal_ml_core import AnnotationTask, Dataset, db
from tests.conftest import TEST_USER_ID

OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"


def _dataset(app, task_type="classification", labels=None, owner_id=TEST_USER_ID, tasks=()):
    """A Dataset with one task per (label, status, source) in `tasks`."""
    with app.app_context():
        dataset = Dataset(
            owner_id=owner_id,
            name="Clauses",
            file_path="/tmp/clauses.csv",
            row_count=len(tasks),
            columns=[],
            task_type=task_type,
            labels=labels,
        )
        db.session.add(dataset)
        db.session.flush()
        for index, (label, status, source) in enumerate(tasks):
            db.session.add(
                AnnotationTask(
                    dataset_id=dataset.id,
                    item_index=index,
                    text=f"clause {index}",
                    annotation_label=label,
                    status=status,
                    label_source=source,
                )
            )
        db.session.commit()
        return dataset.id


def _put(client, auth_headers, dataset_id, body):
    return client.put(f"/api/datasets/{dataset_id}/label-set", json=body, headers=auth_headers)


def _error(response):
    assert response.status_code == 400
    return response.get_json()["error"]


def test_names_labels_on_a_dataset_without_any(client, auth_headers, app):
    dataset_id = _dataset(app)

    response = _put(client, auth_headers, dataset_id, {"labels": [" Obligation ", "Permission", "Prohibition"]})

    assert response.status_code == 200
    assert response.get_json()["dataset"]["labels"] == ["Obligation", "Permission", "Prohibition"]
    shown = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers).get_json()
    assert response.get_json() == shown


@pytest.mark.parametrize("task_type", ["classification", "document_classification", "judgment_prediction"])
def test_every_single_label_task_type_is_accepted(client, auth_headers, app, task_type):
    dataset_id = _dataset(app, task_type=task_type)

    response = _put(client, auth_headers, dataset_id, {"labels": ["Yes", "No"]})

    assert response.status_code == 200


def test_replaces_the_list_freely_before_anything_is_labelled(client, auth_headers, app):
    dataset_id = _dataset(
        app,
        labels=["Obligation", "Permission"],
        tasks=[(None, "pending", None), ("Obligation", "skipped", "human"), ("Permission", "pending", "llm_bulk")],
    )

    response = _put(client, auth_headers, dataset_id, {"labels": ["Duty", "Right"]})

    assert response.status_code == 200
    assert response.get_json()["dataset"]["labels"] == ["Duty", "Right"]


def test_after_labelling_adds_new_labels_and_changes_unused_ones(client, auth_headers, app):
    dataset_id = _dataset(
        app,
        labels=["Obligation", "Permission", "Prohibiton"],
        tasks=[("Obligation", "annotated", "human"), ("Permission", "annotated", "file")],
    )

    response = _put(client, auth_headers, dataset_id, {"labels": ["Permission", "Obligation", "Prohibition", "Other"]})

    assert response.status_code == 200
    assert response.get_json()["dataset"]["labels"] == ["Permission", "Obligation", "Prohibition", "Other"]


def test_after_labelling_refuses_to_drop_a_used_label(client, auth_headers, app):
    dataset_id = _dataset(
        app,
        labels=["Obligation", "Permission"],
        tasks=[("Obligation", "annotated", "llm_confirmed")],
    )

    response = _put(client, auth_headers, dataset_id, {"labels": ["Permission", "Prohibition"]})

    assert _error(response) == "'Obligation' is already used on labelled items, so it can't be renamed or removed."
    with app.app_context():
        assert db.session.get(Dataset, dataset_id).labels == ["Obligation", "Permission"]


def test_after_labelling_refuses_a_rename_even_of_case(client, auth_headers, app):
    dataset_id = _dataset(app, labels=["Obligation", "Permission"], tasks=[("Obligation", "annotated", "human")])

    response = _put(client, auth_headers, dataset_id, {"labels": ["obligation", "Permission"]})

    assert _error(response) == "'Obligation' is already used on labelled items, so it can't be renamed or removed."


def test_labels_typed_before_any_list_must_be_kept(client, auth_headers, app):
    dataset_id = _dataset(app, labels=None, tasks=[("Duty", "annotated", "human")])

    refused = _put(client, auth_headers, dataset_id, {"labels": ["Obligation", "Permission"]})
    accepted = _put(client, auth_headers, dataset_id, {"labels": ["Duty", "Permission"]})

    assert _error(refused) == "'Duty' is already used on labelled items, so it can't be renamed or removed."
    assert accepted.status_code == 200


@pytest.mark.parametrize(
    "labels, message",
    [
        ([], "Use at least two labels."),
        (["Obligation"], "Use at least two labels."),
        (["Obligation", "  "], "Label names can't be empty."),
        (["Obligation", ""], "Label names can't be empty."),
        (["Obligation", "obligation "], "Two labels have the same name."),
        (["Obligation", "Permission", "Obligation"], "Two labels have the same name."),
        ([f"Label {i}" for i in range(51)], "Use at most 50 labels."),
        (["Obligation", "x" * 101], "Keep each label name to 100 characters or fewer."),
    ],
)
def test_refuses_lists_that_do_not_name_two_distinct_labels(client, auth_headers, app, labels, message):
    dataset_id = _dataset(app)

    assert _error(_put(client, auth_headers, dataset_id, {"labels": labels})) == message
    with app.app_context():
        assert db.session.get(Dataset, dataset_id).labels is None


def test_accepts_the_largest_list(client, auth_headers, app):
    dataset_id = _dataset(app)
    labels = [f"{i:03d}" + "x" * 97 for i in range(50)]

    assert _put(client, auth_headers, dataset_id, {"labels": labels}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        None,
        ["Obligation", "Permission"],
        {},
        {"labels": "Obligation, Permission"},
        {"labels": ["Obligation", 2]},
        {"labels": ["Obligation", None]},
        {"labels": ["Obligation", "Permission"], "definitions": {}},
    ],
)
def test_refuses_a_body_that_is_not_a_list_of_names(client, auth_headers, app, body):
    dataset_id = _dataset(app)

    assert _error(_put(client, auth_headers, dataset_id, body)) == "Send labels as a list of names."


def test_refuses_a_body_not_sent_as_json(client, auth_headers, app):
    dataset_id = _dataset(app)

    response = client.put(
        f"/api/datasets/{dataset_id}/label-set", data="labels=Obligation,Permission", headers=auth_headers
    )

    assert _error(response) == "Send labels as a list of names."


@pytest.mark.parametrize("task_type", ["ner", "named_entity_recognition", "summarization", "topic_modeling", "clustering"])
def test_refuses_other_task_types(client, auth_headers, app, task_type):
    dataset_id = _dataset(app, task_type=task_type)

    response = _put(client, auth_headers, dataset_id, {"labels": ["Obligation", "Permission"]})

    assert _error(response) == "Labels can only be set for single-label tasks such as Text Classification."


def test_someone_elses_or_a_missing_dataset_is_not_found(client, auth_headers, app):
    theirs = _dataset(app, owner_id=OTHER_USER_ID)

    for dataset_id in (theirs, "no-such-dataset"):
        response = _put(client, auth_headers, dataset_id, {"labels": ["Obligation", "Permission"]})
        assert response.status_code == 404
    with app.app_context():
        assert db.session.get(Dataset, theirs).labels is None


def test_needs_a_token(client, app):
    dataset_id = _dataset(app)

    response = client.put(f"/api/datasets/{dataset_id}/label-set", json={"labels": ["Obligation", "Permission"]})

    assert response.status_code == 401


def test_named_labels_let_llm_labelling_through(client, auth_headers, app):
    """The reason for the endpoint: LLM labelling refuses a Dataset without
    two named labels, and gets past that check once they are set."""
    dataset_id = _dataset(app, tasks=[(None, "pending", None)])
    refusal = "Define at least two labels for this dataset first."
    estimate = f"/api/llm/datasets/{dataset_id}/annotate/estimate"

    before = client.post(estimate, json={}, headers=auth_headers)
    _put(client, auth_headers, dataset_id, {"labels": ["Obligation", "Permission"]})
    after = client.post(estimate, json={}, headers=auth_headers)

    assert before.get_json()["error"] == refusal
    assert (after.get_json() or {}).get("error") != refusal
