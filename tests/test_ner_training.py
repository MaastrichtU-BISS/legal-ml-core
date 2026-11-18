"""NER Datasets train token classifiers on their saved spans (BAIGLE #117).

The spans are the stored NER entities (character offsets into the raw task
text); training never cleans the text, because cleaning would move the offsets.
"""
import json

import joblib
import pandas as pd
import pytest

from legal_ml_core import Dataset, TrainingJob, db
from legal_ml_core.annotation_queue import create_annotation_tasks
from legal_ml_core.compute.tasks import process_prediction_task, process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer
from legal_ml_core.storage import get_model_path
from tests.conftest import TEST_USER_ID

TEXTS = [
    "John Smith works for Acme Corp.",
    "Maria Garcia sued Acme Corp in Paris.",
    "The court heard John Smith in Paris.",
    "Acme Corp hired Maria Garcia.",
    "Paris is where Acme Corp met John Smith.",
    "Maria Garcia works for Acme Corp in Paris.",
    "The judge praised John Smith.",
    "Acme Corp and John Smith settled in Paris.",
]
ENTITIES = {
    "John Smith": "PERSON",
    "Maria Garcia": "PERSON",
    "Acme Corp": "ORG",
    "Paris": "LOC",
}
ENTITY_TYPES = ["LOC", "ORG", "PERSON"]


def _spans(text):
    spans = []
    for needle, kind in ENTITIES.items():
        start = text.find(needle)
        if start >= 0:
            spans.append({"text": needle, "start": start, "end": start + len(needle), "type": kind})
    return sorted(spans, key=lambda span: span["start"])


def _write_ner_csv(tmp_path, spans=True):
    path = tmp_path / "ner.csv"
    entities = [json.dumps(_spans(text)) for text in TEXTS] if spans else ["PERSON, ORG"] * len(TEXTS)
    pd.DataFrame({"text": TEXTS, "entities": entities}).to_csv(path, index=False)
    return str(path)


def _ner_dataset(app, tmp_path, with_tasks=True, queued=True):
    """An NER Dataset whose file holds spans (``with_tasks``), its rows already
    turned into tasks when ``queued``; an import arrives with ``queued=False``."""
    path = _write_ner_csv(tmp_path, spans=with_tasks)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="NER",
            file_path=path,
            row_count=len(TEXTS),
            columns=[],
            task_type="ner",
        )
        db.session.add(dataset)
        db.session.commit()
        if with_tasks and queued:
            create_annotation_tasks(dataset)
            db.session.commit()
        return dataset.id


def _tiny_bert(path):
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    vocab = path / "vocab.txt"
    words = sorted({word.strip(".,") for text in TEXTS for word in text.split()})
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(path)
    config = BertConfig(
        vocab_size=5 + len(words),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=8,
        num_labels=7,
    )
    BertForTokenClassification(config).save_pretrained(path)
    return str(path)


def _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path, parameters=None):
    created = client.post(
        "/api/models/",
        json={
            "name": "Entities",
            "dataset_id": dataset_id,
            "model_type": "distilbert_ner",
            "parameters": parameters or {"epochs": 1, "batch_size": 4, "max_length": 128},
        },
        headers=auth_headers,
    )
    assert created.status_code == 201, created.get_json()
    model_id = created.get_json()["model"]["id"]
    started = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert started.status_code == 200, started.get_json()

    task = json.loads(fake_redis.rpop("training_queue"))
    task["parameters"]["model_name"] = _tiny_bert(tmp_path)  # stands in for a download
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])
    return model_id, db.session.get(TrainingJob, task["job_id"])


def test_an_ner_dataset_with_imported_spans_trains_end_to_end(
    client, auth_headers, app, fake_redis, tmp_path
):
    # Straight from an import: the spans are in the file and no queue has
    # been read, so no task exists yet.
    dataset_id = _ner_dataset(app, tmp_path, queued=False)

    model_id, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)

    assert job.status == "completed", job.error_message
    assert {"f1_score", "precision", "recall", "classification_report"} <= set(job.metrics)
    assert job.metrics["entity_types"] == ENTITY_TYPES
    assert isinstance(job.metrics["classification_report"], str)

    bundle = joblib.load(get_model_path(app.config["MODEL_FOLDER"], model_id))
    assert bundle["task"] == "ner"
    assert bundle["label_list"][0] == "O"
    assert {"B-PERSON", "I-PERSON", "B-ORG", "I-ORG", "B-LOC", "I-LOC"} == set(bundle["label_list"][1:])
    assert bundle["max_length"] == 128


def test_an_ner_dataset_without_annotated_rows_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = _ner_dataset(app, tmp_path, with_tasks=False)
    message = "Annotate some rows first: this Dataset has no labelled entities to learn from."

    fit = client.get(f"/api/models/available?dataset_id={dataset_id}", headers=auth_headers).get_json()

    assert fit["models"]["distilbert_ner"]["usable"] is False
    assert fit["models"]["distilbert_ner"]["reason"] == message
    assert fit["advice"] == message

    created = client.post(
        "/api/models/",
        json={
            "name": "Entities",
            "dataset_id": dataset_id,
            "model_type": "distilbert_ner",
            "parameters": {},
        },
        headers=auth_headers,
    )
    assert created.status_code == 400
    assert created.get_json()["error"] == message


def test_prediction_returns_entity_spans_with_offsets(
    client, auth_headers, app, fake_redis, tmp_path
):
    dataset_id = _ner_dataset(app, tmp_path)
    model_id, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)
    assert job.status == "completed", job.error_message

    text = "John Smith works for Acme Corp in Paris."
    result = process_prediction_task(
        {
            "prediction_id": "p1",
            "model_id": model_id,
            "model_type": "distilbert_ner",
            "texts": [text],
        },
        app.config["MODEL_FOLDER"],
    )

    entry = result["predictions"][0]
    assert entry["text"] == text
    assert isinstance(entry["prediction"], str)
    for entity in entry["entities"]:
        assert 0 <= entity["start"] < entity["end"] <= len(text)
        assert text[entity["start"]:entity["end"]] == entity["text"]
        assert entity["type"] in ENTITY_TYPES
        assert isinstance(entity["score"], float)
    assert entry["prediction"] == ", ".join(
        dict.fromkeys(entity["type"] for entity in entry["entities"])
    )


class _NullRedis:
    def setex(self, *a, **k):
        pass


def test_prediction_merges_label_runs_into_spans(app, tmp_path):
    """Every word labelled B-PER becomes its own span, with offsets into the text."""
    import torch
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    words = ["John", "Smith", "sued", "Acme", "Corp"]
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))

    config = BertConfig(
        vocab_size=5 + len(words),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=8,
        num_labels=3,
    )
    model = BertForTokenClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))

    trainer = MLModelTrainer(redis_client=_NullRedis())
    trainer.model = model
    trainer.tokenizer = tokenizer
    trainer.task = "ner"
    trainer.label_list = ["O", "B-PER", "I-PER"]
    trainer.max_length = 64
    trainer.save_model(get_model_path(str(tmp_path), "ner-predict"))

    text = "John Smith sued Acme Corp"
    result = process_prediction_task(
        {"prediction_id": "p", "model_id": "ner-predict", "model_type": "legalbert_ner", "texts": [text]},
        str(tmp_path),
    )

    entry = result["predictions"][0]
    assert entry["entities"]
    assert entry["prediction"] == "PER"
    for entity in entry["entities"]:
        assert text[entity["start"]:entity["end"]] == entity["text"]
        assert entity["type"] == "PER"


def test_an_entity_model_cannot_be_exported(client, auth_headers, app, fake_redis, tmp_path):
    from legal_ml_core.compute.export import process_export_task

    dataset_id = _ner_dataset(app, tmp_path)
    model_id, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)
    assert job.status == "completed", job.error_message

    with pytest.raises(ValueError, match="Only classification Models"):
        process_export_task(
            {"model_id": model_id, "model_type": "distilbert_ner", "texts": ["John Smith"]},
            app.config["MODEL_FOLDER"],
        )
