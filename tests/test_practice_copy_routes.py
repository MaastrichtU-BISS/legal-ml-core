"""POST /api/datasets/<id>/practice-copy makes an unlabelled copy of a labelled
single-label Dataset (BAIGLE #197)."""

import os

import pandas as pd
import pytest

from legal_ml_core import AnnotationTask, Dataset, db
from legal_ml_core.annotation_queue import create_annotation_tasks
from legal_ml_core.models import LLMPrompt
from tests.conftest import TEST_USER_ID

OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"
LABELS = ["Obligation", "Permission", "Prohibition"]
ROWS = [
    ("The provider shall keep records.", "Obligation"),
    ("The provider may appoint an agent.", "Permission"),
    ("The provider shall not sell the data.", "Prohibition"),
    ("Member States shall report yearly.", "Obligation"),
]


def _headers_for(sub):
    import json

    claims = {"sub": sub, "iss": "https://keycloak.test/realms/biss", "aud": "baigle"}
    return {"Authorization": f"Bearer test-token::{json.dumps(claims)}"}


def _write(app, frame, name="clauses.csv"):
    path = os.path.join(app.config["UPLOAD_FOLDER"], name)
    frame.to_csv(path, index=False)
    return path


def _dataset(
    app,
    rows=ROWS,
    task_type="classification",
    labels=LABELS,
    owner_id=TEST_USER_ID,
    label_column="label",
    extra=None,
    name="Clauses",
    project_id="proj-1",
):
    """A Dataset whose file has `rows` as (text, label); a None label is blank."""
    frame = pd.DataFrame(
        {"text": [text for text, _ in rows], label_column: [label or "" for _, label in rows], **(extra or {})}
    )
    with app.app_context():
        dataset = Dataset(
            owner_id=owner_id,
            name=name,
            description="EU regulation sentences",
            file_path=_write(app, frame, f"{name[:20]}-{task_type}.csv"),
            row_count=len(rows),
            columns=[],
            task_type=task_type,
            labels=labels,
            project_id=project_id,
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def _post(client, headers, dataset_id):
    return client.post(f"/api/datasets/{dataset_id}/practice-copy", headers=headers)


def _copy(client, auth_headers, dataset_id):
    response = _post(client, auth_headers, dataset_id)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["dataset"]


def _add_prompt(app, dataset_id, version, definitions, template="Label it: {labels}\n{text}"):
    with app.app_context():
        db.session.add(LLMPrompt(dataset_id=dataset_id, version=version, template=template, definitions=definitions))
        db.session.commit()


def _snapshot(app, dataset_id):
    """Everything about a Dataset a copy could change."""
    with app.app_context():
        dataset = db.session.get(Dataset, dataset_id)
        tasks = sorted(
            (t.item_index, t.text, t.annotation_label, t.status, t.label_source, t.file_label)
            for t in AnnotationTask.query.filter_by(dataset_id=dataset_id)
        )
        prompts = sorted(
            (p.version, p.template, str(p.definitions)) for p in LLMPrompt.query.filter_by(dataset_id=dataset_id)
        )
        with open(dataset.file_path, "rb") as f:
            content = f.read()
        return dataset.to_dict(), tasks, prompts, content


def test_copy_is_owned_by_the_same_user_and_named_practice(client, auth_headers, app):
    dataset_id = _dataset(app)

    copy = _copy(client, auth_headers, dataset_id)

    assert copy["id"] != dataset_id
    assert copy["owner_id"] == TEST_USER_ID
    assert copy["name"] == "Clauses (practice)"
    assert copy["task_type"] == "classification"
    assert copy["project_id"] == "proj-1"
    shown = client.get(f"/api/datasets/{copy['id']}", headers=auth_headers).get_json()
    assert shown == {"dataset": copy}


def test_copy_records_its_origin(client, auth_headers, app):
    dataset_id = _dataset(app)

    copy = _copy(client, auth_headers, dataset_id)

    assert copy["origin_dataset_id"] == dataset_id
    original = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers).get_json()["dataset"]
    assert original["origin_dataset_id"] is None


def test_copy_has_the_same_text_and_no_labels(client, auth_headers, app):
    # A second column naming the classes must not carry the labels over either.
    dataset_id = _dataset(app, extra={"category": [label for _, label in ROWS], "source": ["a", "b", "c", "d"]})
    with app.app_context():
        # A person's label on the original, and a file label they changed.
        db.session.add(
            AnnotationTask(
                dataset_id=dataset_id, item_index=0, text=ROWS[0][0], annotation_label="Permission",
                status="annotated", label_source="human", file_label="Obligation",
            )
        )
        db.session.commit()

    copy = _copy(client, auth_headers, dataset_id)

    assert copy["annotated_count"] == 0
    assert copy["total_count"] == len(ROWS)
    with app.app_context():
        stored = db.session.get(Dataset, copy["id"])
        frame = pd.read_csv(stored.file_path)
        tasks = AnnotationTask.query.filter_by(dataset_id=copy["id"]).order_by(AnnotationTask.item_index).all()
    assert list(frame.columns) == ["text", "label"]
    assert frame["text"].tolist() == [text for text, _ in ROWS]
    assert frame["label"].isna().all()
    assert [t.text for t in tasks] == [text for text, _ in ROWS]
    assert {(t.status, t.annotation_label, t.label_source, t.file_label) for t in tasks} == {
        ("pending", None, None, None)
    }


def test_copy_of_a_judgment_dataset_blanks_the_judgment(client, auth_headers, app):
    rows = [("Facts one.", "guilty"), ("Facts two.", "not_guilty")]
    dataset_id = _dataset(
        app, rows=rows, task_type="judgment_prediction", labels=["guilty", "not_guilty"], label_column="judgment"
    )

    copy = _copy(client, auth_headers, dataset_id)

    with app.app_context():
        frame = pd.read_csv(db.session.get(Dataset, copy["id"]).file_path)
    assert list(frame.columns) == ["text", "judgment"]
    assert frame["judgment"].isna().all()
    assert copy["annotated_count"] == 0


def test_copy_keeps_the_label_list(client, auth_headers, app):
    dataset_id = _dataset(app)

    copy = _copy(client, auth_headers, dataset_id)

    assert copy["labels"] == LABELS


def test_copy_without_a_named_list_takes_the_labels_its_items_have(client, auth_headers, app):
    dataset_id = _dataset(app, labels=None)

    copy = _copy(client, auth_headers, dataset_id)

    assert copy["labels"] == LABELS


def test_numeric_labels_with_a_blank_cell_are_listed_once_each(client, auth_headers, app):
    # A 0/1 column with a blank cell is read as floats; the tasks store "1" and "0".
    frame = pd.DataFrame({"text": ["Appeal one.", "Appeal two.", "Appeal three.", "Appeal four."], "judgment": [1, 0, None, 1]})
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID, name="Judgments", description="", file_path=_write(app, frame, "judgments.csv"),
            row_count=4, columns=[], task_type="judgment_prediction", labels=None, project_id="proj-1",
        )
        db.session.add(dataset)
        db.session.commit()
        create_annotation_tasks(dataset)
        db.session.commit()
        dataset_id = dataset.id

    copy = _copy(client, auth_headers, dataset_id)

    assert sorted(copy["labels"]) == ["0", "1"]


def test_a_custom_text_and_label_column_are_read_and_no_other_column_carries_labels(client, auth_headers, app):
    frame = pd.DataFrame(
        {
            "body": ["The provider shall keep records.", "The provider may appoint an agent."],
            "text": ["unrelated", "unrelated"],
            "category": ["Obligation", "Permission"],
            "label": ["Obligation", "Permission"],
        }
    )
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID, name="Custom", description="", file_path=_write(app, frame, "custom.csv"),
            row_count=2, columns=[], task_type="classification", labels=["Obligation", "Permission"],
            project_id="proj-1", text_column="body", label_column="category",
        )
        db.session.add(dataset)
        db.session.commit()
        dataset_id = dataset.id

    copy = _copy(client, auth_headers, dataset_id)

    with app.app_context():
        written = pd.read_csv(db.session.get(Dataset, copy["id"]).file_path, keep_default_na=False)
    assert list(written["text"]) == ["The provider shall keep records.", "The provider may appoint an agent."]
    assert {value for column in written.columns if column != "text" for value in written[column]} <= {""}


def test_copy_gets_the_current_prompt_definitions_as_a_new_version(client, auth_headers, app):
    dataset_id = _dataset(app)
    _add_prompt(app, dataset_id, 1, {"Obligation": "old words"})
    current = {
        "Obligation": "Someone must do something.",
        "Permission": "Someone may do something.",
        "Prohibition": "Someone must not do something.",
    }
    _add_prompt(app, dataset_id, 2, current, template="Current: {labels}\n{text}")

    copy = _copy(client, auth_headers, dataset_id)

    prompt = client.get(f"/api/llm/datasets/{copy['id']}/prompt", headers=auth_headers).get_json()
    assert prompt == {"version": 1, "template": "Current: {labels}\n{text}", "definitions": current}


def test_copy_of_a_dataset_on_the_default_prompt_stays_on_it(client, auth_headers, app):
    dataset_id = _dataset(app)

    copy = _copy(client, auth_headers, dataset_id)

    original = client.get(f"/api/llm/datasets/{dataset_id}/prompt", headers=auth_headers).get_json()
    prompt = client.get(f"/api/llm/datasets/{copy['id']}/prompt", headers=auth_headers).get_json()
    assert prompt == original
    assert prompt["version"] == 0


def test_the_original_is_unchanged(client, auth_headers, app):
    dataset_id = _dataset(app)
    _add_prompt(app, dataset_id, 1, {"Obligation": "Someone must do something."})
    with app.app_context():
        db.session.add(
            AnnotationTask(
                dataset_id=dataset_id, item_index=1, text=ROWS[1][0], annotation_label="Permission",
                status="annotated", label_source="human",
            )
        )
        db.session.commit()
    before = _snapshot(app, dataset_id)

    _copy(client, auth_headers, dataset_id)

    assert _snapshot(app, dataset_id) == before


def test_a_long_name_is_cut_to_fit(client, auth_headers, app):
    dataset_id = _dataset(app, name="x" * 255)

    copy = _copy(client, auth_headers, dataset_id)

    assert len(copy["name"]) == 255
    assert copy["name"].endswith("x (practice)")


@pytest.mark.parametrize("task_type", ["ner", "summarization", "topic_modeling"])
def test_other_task_types_are_refused(client, auth_headers, app, task_type):
    dataset_id = _dataset(app, task_type=task_type)

    response = _post(client, auth_headers, dataset_id)

    assert response.status_code == 400
    assert response.get_json()["error"] == (
        "Practice copies can only be made of single-label tasks such as Text Classification."
    )
    with app.app_context():
        assert Dataset.query.count() == 1


def test_an_unlabelled_dataset_is_refused(client, auth_headers, app):
    dataset_id = _dataset(app, rows=[(text, None) for text, _ in ROWS])

    response = _post(client, auth_headers, dataset_id)

    assert response.status_code == 400
    assert response.get_json()["error"] == "This dataset has no labels yet. Label it directly instead."
    with app.app_context():
        assert Dataset.query.count() == 1
        assert os.listdir(app.config["UPLOAD_FOLDER"]) == ["Clauses-classification.csv"]


def test_a_dataset_labelled_only_by_a_person_can_be_copied(client, auth_headers, app):
    dataset_id = _dataset(app, rows=[(text, None) for text, _ in ROWS])
    with app.app_context():
        db.session.add(
            AnnotationTask(
                dataset_id=dataset_id, item_index=0, text=ROWS[0][0], annotation_label="Obligation",
                status="annotated", label_source="human",
            )
        )
        db.session.commit()

    copy = _copy(client, auth_headers, dataset_id)

    assert copy["annotated_count"] == 0


def test_another_users_dataset_is_not_found(client, app):
    dataset_id = _dataset(app, owner_id=OTHER_USER_ID)

    response = _post(client, _headers_for(TEST_USER_ID), dataset_id)

    assert response.status_code == 404
    with app.app_context():
        assert Dataset.query.count() == 1


def test_a_missing_dataset_is_not_found(client, auth_headers):
    response = _post(client, auth_headers, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


def test_requires_sign_in(client, app):
    dataset_id = _dataset(app)

    response = client.post(f"/api/datasets/{dataset_id}/practice-copy")

    assert response.status_code == 401
