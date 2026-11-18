"""NER Datasets train token classifiers on their saved spans (BAIGLE #117).

The spans are the stored NER entities (character offsets into the raw task
text); training never cleans the text, because cleaning would move the offsets.
"""
import json

import joblib
import pandas as pd
import pytest

from legal_ml_core import AnnotationTask, Dataset, Model, TrainingJob, db
from legal_ml_core.annotation_queue import create_annotation_tasks
from legal_ml_core.compute.tasks import merge_word_spans, process_prediction_task, process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer, word_labels
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


def _write_ner_csv(tmp_path, texts, spans=True):
    path = tmp_path / "ner.csv"
    entities = [json.dumps(_spans(text)) for text in texts] if spans else ["PERSON, ORG"] * len(texts)
    pd.DataFrame({"text": texts, "entities": entities}).to_csv(path, index=False)
    return str(path)


def _ner_dataset_from(app, tmp_path, texts, queued=True):
    """An NER Dataset over ``texts`` with its file's spans, rows already turned
    into tasks when ``queued``; an import arrives with ``queued=False``."""
    path = _write_ner_csv(tmp_path, texts)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="NER",
            file_path=path,
            row_count=len(texts),
            columns=[],
            task_type="ner",
        )
        db.session.add(dataset)
        db.session.commit()
        if queued:
            create_annotation_tasks(dataset)
            db.session.commit()
        return dataset.id


def _ner_dataset(app, tmp_path, with_tasks=True, queued=True):
    """An NER Dataset whose file holds spans (``with_tasks``), its rows already
    turned into tasks when ``queued``; an import arrives with ``queued=False``."""
    path = _write_ner_csv(tmp_path, TEXTS, spans=with_tasks)
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


def _run_ner(app, fake_redis, dataset_id, tmp_path, parameters):
    """Trains through the worker on a hand-made Model and job, the way a run
    reaches the worker: no route, so the catalogue's parameter bounds do not
    apply and a small max_length can be tested."""
    with app.app_context():
        dataset = db.session.get(Dataset, dataset_id)
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset_id,
            name="Entities",
            model_type="distilbert_ner",
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
                "model_type": "distilbert_ner",
                "parameters": {**parameters, "model_name": _tiny_bert(tmp_path)},
                "dataset_path": dataset.file_path,
                "task_type": "ner",
                "preprocessing_options": {},
            },
            fake_redis,
            app.config["MODEL_FOLDER"],
        )
        return model.id, db.session.get(TrainingJob, job.id)


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


def test_merge_takes_a_word_from_its_first_sub_token():
    """A word split into ## pieces keeps one label: its first sub-token's, with
    the span reaching the end of the word, whatever the pieces predicted."""
    spans = merge_word_spans(
        "smithson sued",
        word_ids=[None, 0, 0, 1, None],
        offsets=[(0, 0), (0, 5), (5, 8), (9, 13), (0, 0)],
        labels=["O", "B-PER", "I-PER", "O", "O"],
        scores=[0.1, 0.9, 0.2, 0.1, 0.1],
    )

    assert spans == [{"start": 0, "end": 8, "type": "PER", "scores": [0.9]}]


def test_merge_joins_and_restarts_spans():
    spans = merge_word_spans(
        "a b c d",
        word_ids=[0, 1, 2, 3],
        offsets=[(0, 1), (2, 3), (4, 5), (6, 7)],
        labels=["B-X", "I-X", "B-X", "I-X"],
        scores=[0.5, 0.6, 0.7, 0.8],
    )

    assert spans == [
        {"start": 0, "end": 3, "type": "X", "scores": [0.5, 0.6]},
        {"start": 4, "end": 7, "type": "X", "scores": [0.7, 0.8]},
    ]


def test_merge_starts_a_span_on_a_lone_i_label():
    spans = merge_word_spans(
        "a b",
        word_ids=[0, 1],
        offsets=[(0, 1), (2, 3)],
        labels=["I-X", "O"],
        scores=[0.4, 0.1],
    )

    assert spans == [{"start": 0, "end": 1, "type": "X", "scores": [0.4]}]


def test_merge_stitches_a_word_that_two_windows_saw():
    """The word repeats across overlapping windows: its label comes from the
    first window that saw it, its span from the widest view of it."""
    spans = merge_word_spans(
        "smithson",
        word_ids=[0, 0, 0],
        offsets=[(0, 5), (0, 5), (5, 8)],
        labels=["B-PER", "B-PER", "I-PER"],
        scores=[0.9, 0.8, 0.7],
    )

    assert spans == [{"start": 0, "end": 8, "type": "PER", "scores": [0.9]}]


def test_word_labels_b_and_i_at_word_boundaries():
    assert word_labels([(0, 4), (5, 10), (11, 16)], [{"start": 0, "end": 10, "type": "PERSON"}]) == [
        "B-PERSON",
        "I-PERSON",
        "O",
    ]


def test_word_labels_starts_each_of_two_adjacent_spans_of_one_type():
    spans = [{"start": 0, "end": 5, "type": "PERSON"}, {"start": 6, "end": 11, "type": "PERSON"}]
    assert word_labels([(0, 5), (6, 11)], spans) == ["B-PERSON", "B-PERSON"]


def test_word_labels_counts_a_word_a_span_starts_inside():
    # The span starts mid-word: the whole word is in it.
    assert word_labels([(0, 8), (9, 15)], [{"start": 3, "end": 12, "type": "ORG"}]) == ["B-ORG", "I-ORG"]


def test_word_labels_counts_a_word_a_span_starts_before_on_a_space():
    # The span starts on the space before the word.
    assert word_labels([(0, 4), (6, 11)], [{"start": 5, "end": 11, "type": "ORG"}]) == ["O", "B-ORG"]


def test_word_labels_keeps_punctuation_inside_a_span():
    # "Acme Corp." — the full stop rides on the last word.
    assert word_labels([(0, 4), (5, 10)], [{"start": 0, "end": 10, "type": "ORG"}]) == ["B-ORG", "I-ORG"]


def test_word_labels_ignores_a_span_past_the_end():
    assert word_labels([(0, 4), (5, 10)], [{"start": 12, "end": 16, "type": "ORG"}]) == ["O", "O"]


def test_prediction_joins_a_word_split_into_sub_word_pieces(app, tmp_path):
    """A word the vocabulary only holds in pieces ("smith" + "##son") comes back
    as one span over the whole word, not over its first piece."""
    import torch
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    vocab = tmp_path / "pieces.txt"
    words = ["john", "smith", "##son", "sued", "acme", "corp"]
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "", *words]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    assert tokenizer.tokenize("smithson") == ["smith", "##son"]

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
        model.classifier.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))  # every token B-PER

    trainer = MLModelTrainer(redis_client=_NullRedis())
    trainer.model = model
    trainer.tokenizer = tokenizer
    trainer.task = "ner"
    trainer.label_list = ["O", "B-PER", "I-PER"]
    trainer.max_length = 64
    trainer.save_model(get_model_path(str(tmp_path), "ner-pieces"))

    text = "john smithson sued acme corp"
    result = process_prediction_task(
        {"prediction_id": "p", "model_id": "ner-pieces", "model_type": "legalbert_ner", "texts": [text]},
        str(tmp_path),
    )

    entry = result["predictions"][0]
    texts = [entity["text"] for entity in entry["entities"]]
    assert "smithson" in texts
    assert "smith" not in texts
    start = text.index("smithson")
    piece = next(entity for entity in entry["entities"] if entity["text"] == "smithson")
    assert (piece["start"], piece["end"]) == (start, start + len("smithson"))
    assert entry["prediction"] == "PER"


def test_a_text_longer_than_max_length_trains_and_predicts_to_its_end(app, fake_redis, tmp_path):
    """Long texts train in overlapping windows and are predicted the same way:
    the words at the end are inside the last window, with exact offsets."""
    import torch

    long_text = " ".join(
        ["John Smith works for Acme Corp in Paris"] * 5 + ["The court heard John Smith in Paris"]
    )
    texts = TEXTS + [long_text]
    path = tmp_path / "long.csv"
    pd.DataFrame({"text": texts, "entities": [json.dumps(_spans(text)) for text in texts]}).to_csv(
        path, index=False
    )
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="Long",
            file_path=str(path),
            row_count=len(texts),
            columns=[],
            task_type="ner",
        )
        db.session.add(dataset)
        db.session.commit()
        create_annotation_tasks(dataset)
        db.session.commit()
        dataset_id = dataset.id

    model_id, job = _run_ner(
        app, fake_redis, dataset_id, tmp_path, {"epochs": 1, "batch_size": 2, "max_length": 16}
    )
    assert job.status == "completed", job.error_message
    assert {"f1_score", "precision", "recall", "classification_report"} <= set(job.metrics)

    # Make the prediction deterministic: every token a B-PERSON, so every word
    # is its own span. Cut at max_length tokens, the words at the end would
    # never come back at all.
    bundle_path = get_model_path(app.config["MODEL_FOLDER"], model_id)
    bundle = joblib.load(bundle_path)
    with torch.no_grad():
        bundle["model"].classifier.weight.zero_()
        bias = torch.zeros(len(bundle["label_list"]))
        bias[bundle["label_list"].index("B-PERSON")] = 10.0
        bundle["model"].classifier.bias.copy_(bias)
    joblib.dump(bundle, bundle_path)

    result = process_prediction_task(
        {
            "prediction_id": "p",
            "model_id": model_id,
            "model_type": "distilbert_ner",
            "texts": [long_text],
        },
        app.config["MODEL_FOLDER"],
    )

    entry = result["predictions"][0]
    assert entry["text"] == long_text
    for entity in entry["entities"]:
        assert 0 <= entity["start"] < entity["end"] <= len(long_text)
        assert long_text[entity["start"]:entity["end"]] == entity["text"]
    end = long_text.rfind("Paris")
    last = next(entity for entity in entry["entities"] if entity["start"] == end)
    assert (last["start"], last["end"]) == (end, end + len("Paris"))


def test_asking_what_an_import_can_train_writes_nothing(client, auth_headers, app, tmp_path):
    # Straight from an import: the spans are in the file and no task exists.
    dataset_id = _ner_dataset(app, tmp_path, queued=False)

    fit = client.get(
        f"/api/models/available?dataset_id={dataset_id}", headers=auth_headers
    ).get_json()

    assert fit["models"]["legalbert_ner"]["usable"] is True
    assert fit["models"]["distilbert_ner"]["usable"] is True
    assert fit["advice"] is None
    with app.app_context():
        assert AnnotationTask.query.count() == 0


def test_a_dataset_whose_rows_have_no_entities_fails_the_run(
    client, auth_headers, app, fake_redis, tmp_path
):
    # Every row annotated, but [] everywhere: nothing to learn from.
    texts = TEXTS[:4]
    path = tmp_path / "empty.csv"
    pd.DataFrame({"text": texts, "entities": ["[]"] * len(texts)}).to_csv(path, index=False)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="Empty",
            file_path=str(path),
            row_count=len(texts),
            columns=[],
            task_type="ner",
        )
        db.session.add(dataset)
        db.session.commit()
        create_annotation_tasks(dataset)
        db.session.commit()
        dataset_id = dataset.id

    _, job = _run_ner(app, fake_redis, dataset_id, tmp_path, {"epochs": 1, "max_length": 64})

    assert job.status == "failed"
    assert job.error_message == "This Dataset has no labelled entities to learn from."


def test_a_run_whose_eval_split_holds_only_empty_rows_completes(app, tmp_path):
    """Two examples, the entity-less one last: the split (seed 42) makes it the
    whole evaluation split, and seqeval must not crash on its empty gold."""
    import torch
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    path = tmp_path / "tiny-bert"
    path.mkdir()
    words = sorted({word.strip(".,") for text in TEXTS for word in text.split()})
    (path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "", *words]))
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    config = BertConfig(
        vocab_size=5 + len(words),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=8,
        num_labels=7,
    )
    BertForTokenClassification(config).save_pretrained(path)

    trainer = MLModelTrainer(redis_client=_NullRedis())
    metrics = trainer.train_token_classification(
        [(TEXTS[0], _spans(TEXTS[0])), (TEXTS[1], [])],
        "distilbert_ner",
        {
            "model_name": str(path),
            "epochs": 1,
            "batch_size": 2,
            "max_length": 64,
            "test_size": 0.5,
            "model_id": "eval-empty",
            "output_dir": str(tmp_path / "transformer_eval-empty"),
        },
    )

    assert {"f1_score", "precision", "recall", "classification_report"} <= set(metrics)
    assert isinstance(metrics["classification_report"], str)


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
