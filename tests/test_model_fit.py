"""Which models can train a Dataset, and why not (BAIGLE: model choice by data).

Decided from the task type and the label counts stored with every Dataset, so
no file is read — except an NER Dataset's, only while its rows are not all
queued yet, to see an import's spans. Labels that are nearly all different (a
question/answer set uploaded as classification) are not categories: no
classifier can learn them.
"""
import pandas as pd

from legal_ml_core import AnnotationTask, Dataset, db
from legal_ml_core.annotation_queue import create_annotation_tasks
from tests.conftest import TEST_USER_ID

QA = [
    {"name": "text", "type": "object", "null_count": 0, "unique_count": 1000},
    {"name": "label", "type": "object", "null_count": 0, "unique_count": 1000},
]
DEONTIC = [
    {"name": "text", "type": "object", "null_count": 0, "unique_count": 567},
    {"name": "label", "type": "object", "null_count": 0, "unique_count": 5},
]
CLASSIFIERS = {"svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression", "legalbert", "roberta"}
ENTITY_MODELS = {"legalbert_ner", "distilbert_ner"}
NO_ENTITIES_ADVICE = "Annotate some rows first: this Dataset has no labelled entities to learn from."


def _dataset(app, columns, task_type="classification", rows=None):
    rows = rows if rows is not None else columns[0]["unique_count"] if columns else 10
    with app.app_context():
        dataset = Dataset(owner_id=TEST_USER_ID, name="D", file_path="/tmp/d.csv", row_count=rows, columns=columns, task_type=task_type)
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def _fit(client, auth_headers, dataset_id):
    resp = client.get(f"/api/models/available?dataset_id={dataset_id}", headers=auth_headers)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def test_answers_as_labels_fit_no_classifier_and_say_why(client, auth_headers, app):
    body = _fit(client, auth_headers, _dataset(app, QA))

    models = body["models"]
    assert {t for t, m in models.items() if m["usable"]} == set()
    assert models["svm"]["reason"] == (
        "Support Vector Machine learns from labels that are categories, but this Dataset has "
        "1,000 different labels in 1,000 labelled rows: fewer than 2 examples each."
    )
    assert models["kmeans"]["reason"] == "K-Means groups texts without labels: it trains on Clustering or Topic modelling Datasets."
    assert body["advice"] == (
        "These labels look like answers or free text, not categories. Give the rows a few "
        "categories (annotate them, or re-label the file) to train a model on them."
    )


def test_labels_that_are_categories_fit_every_classifier(client, auth_headers, app):
    body = _fit(client, auth_headers, _dataset(app, DEONTIC))

    usable = {t for t, m in body["models"].items() if m["usable"]}
    assert usable == CLASSIFIERS
    assert all(body["models"][t]["reason"] is None for t in CLASSIFIERS)
    assert body["advice"] is None
    # Every model is listed, so the wizard can show why the rest do not fit.
    assert {"kmeans", "lda", "dbscan"} <= set(body["models"])


def test_unlabelled_rows_do_not_count_against_the_labels(client, auth_headers, app):
    half = [QA[0], {"name": "label", "type": "object", "null_count": 990, "unique_count": 3}]
    body = _fit(client, auth_headers, _dataset(app, half, rows=1000))

    assert body["models"]["svm"]["usable"] is True


def test_a_task_with_no_trainable_models_says_so_for_every_model(client, auth_headers, app):
    body = _fit(client, auth_headers, _dataset(app, [], task_type="summarization"))

    assert not any(m["usable"] for m in body["models"].values())
    assert body["advice"] == "Training for Text Summarization Datasets is not available yet."


def test_an_ner_dataset_without_annotated_rows_says_to_annotate(client, auth_headers, app):
    body = _fit(client, auth_headers, _dataset(app, [], task_type="ner", rows=5))

    assert not any(m["usable"] for m in body["models"].values())
    assert body["advice"] == NO_ENTITIES_ADVICE
    assert {body["models"][model_type]["reason"] for model_type in ENTITY_MODELS} == {NO_ENTITIES_ADVICE}


def test_a_dataset_whose_annotated_rows_hold_no_entities_is_refused(client, auth_headers, app, tmp_path):
    # Every row is annotated, but [] is a row with no entities: nothing to learn.
    path = tmp_path / "empty.csv"
    pd.DataFrame({"text": ["Smith sued Acme.", "Acme paid."], "entities": ["[]", "[]"]}).to_csv(
        path, index=False
    )
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="NER",
            file_path=str(path),
            row_count=2,
            columns=[],
            task_type="ner",
        )
        db.session.add(dataset)
        db.session.commit()
        create_annotation_tasks(dataset)
        db.session.commit()
        dataset_id = dataset.id

    body = _fit(client, auth_headers, dataset_id)

    assert body["models"]["distilbert_ner"]["usable"] is False
    assert body["models"]["distilbert_ner"]["reason"] == NO_ENTITIES_ADVICE


def test_an_annotated_ner_dataset_fits_both_entity_models(client, auth_headers, app):
    dataset_id = _dataset(app, [], task_type="ner", rows=5)
    with app.app_context():
        db.session.add(
            AnnotationTask(
                dataset_id=dataset_id,
                item_index=0,
                text="John Smith sued Acme Corp.",
                status="annotated",
                annotation_label='[{"text": "John Smith", "start": 0, "end": 10, "type": "PERSON"}]',
            )
        )
        db.session.commit()

    body = _fit(client, auth_headers, dataset_id)

    assert {t for t, m in body["models"].items() if m["usable"]} == ENTITY_MODELS
    assert body["advice"] is None


def test_an_ner_model_refuses_a_classification_dataset_with_a_plain_reason(client, auth_headers, app):
    body = _fit(client, auth_headers, _dataset(app, DEONTIC))

    assert body["models"]["distilbert_ner"]["reason"] == (
        "DistilBERT (entities) finds named entities: it trains on Named Entity Recognition Datasets."
    )
    assert body["models"]["distilbert_ner"]["usable"] is False


def test_another_users_dataset_is_not_found(client, auth_headers, app):
    with app.app_context():
        other = Dataset(owner_id="someone-else", name="X", file_path="/tmp/x.csv", row_count=1, columns=[], task_type="classification")
        db.session.add(other)
        db.session.commit()
        other_id = other.id

    assert client.get(f"/api/models/available?dataset_id={other_id}", headers=auth_headers).status_code == 404


def test_creating_a_model_that_cannot_learn_the_labels_is_refused_with_the_same_reason(client, auth_headers, app):
    dataset_id = _dataset(app, QA)

    resp = client.post(
        "/api/models/",
        json={"name": "QA SVM", "dataset_id": dataset_id, "model_type": "svm", "parameters": {}},
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"].startswith("Support Vector Machine learns from labels that are categories")


def test_columns_stored_as_bare_names_fit_by_task_type_alone(client, auth_headers, app):
    # Older Datasets store only column names, no counts: nothing says the labels
    # are not categories, so every classifier stays on offer.
    body = _fit(client, auth_headers, _dataset(app, ["text", "label"], rows=10))

    assert {t for t, m in body["models"].items() if m["usable"]} == CLASSIFIERS
