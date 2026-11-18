"""Summarization Datasets train a text-to-text model on their text and summary
columns (BAIGLE #117, #21).

Nothing downloads: a tiny BART and a character-level byte-BPE tokenizer are
written to disk and stand in for t5-small (``AutoModelForSeq2SeqLM`` loads
either).
"""
import inspect
import json

import joblib
import pandas as pd
import pytest

from legal_ml_core import Dataset, TrainingJob, db
from legal_ml_core.compute.tasks import process_prediction_task, process_training_task
from legal_ml_core.storage import get_model_path
from tests.conftest import TEST_USER_ID

ROWS = [
    ("The court held that the clause was void.", "Clause void."),
    ("The appeal was dismissed with costs.", "Appeal dismissed."),
    ("The tenant must pay the rent on time.", "Pay rent on time."),
    ("The contract ends after two years.", "Ends in two years."),
    ("The judge awarded damages to Smith.", "Damages to Smith."),
]
PARAMETERS = {"epochs": 1, "batch_size": 2, "max_source_length": 128, "max_target_length": 32}


def _tiny_seq2seq(path):
    """A one-layer BART with a character-level byte-BPE tokenizer (no merges),
    saved under ``path``."""
    from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast

    specials = ["<s>", "<pad>", "</s>", "<unk>"]
    # Every printable ASCII character, with "Ġ" for a space as byte-level BPE writes it.
    characters = ["Ġ"] + [chr(code) for code in range(33, 127)]
    vocab = {token: index for index, token in enumerate(specials + characters + ["<mask>"])}
    (path / "vocab.json").write_text(json.dumps(vocab))
    (path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = BartTokenizerFast(vocab_file=str(path / "vocab.json"), merges_file=str(path / "merges.txt"))
    tokenizer.save_pretrained(path)
    config = BartConfig(
        vocab_size=len(vocab),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=256,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
        decoder_start_token_id=vocab["</s>"],
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    )
    BartForConditionalGeneration(config).save_pretrained(path)
    return str(path)


def _summary_dataset(app, tmp_path, rows):
    path = tmp_path / "sums.csv"
    pd.DataFrame({"text": [t for t, _ in rows], "summary": [s for _, s in rows]}).to_csv(path, index=False)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="Summaries",
            file_path=str(path),
            row_count=len(rows),
            columns=[{"name": "text"}, {"name": "summary"}],
            task_type="summarization",
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def _create(client, auth_headers, dataset_id):
    return client.post(
        "/api/models/",
        json={"name": "Summaries", "dataset_id": dataset_id, "model_type": "t5_small", "parameters": PARAMETERS},
        headers=auth_headers,
    )


def _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path):
    created = _create(client, auth_headers, dataset_id)
    assert created.status_code == 201, created.get_json()
    model_id = created.get_json()["model"]["id"]
    started = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert started.status_code == 200, started.get_json()

    task = json.loads(fake_redis.rpop("training_queue"))
    model_dir = tmp_path / "tiny-seq2seq"
    model_dir.mkdir(exist_ok=True)
    task["parameters"]["model_name"] = _tiny_seq2seq(model_dir)  # stands in for a download
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])
    return model_id, db.session.get(TrainingJob, task["job_id"])


def test_a_summarization_dataset_trains_and_reports_rouge(client, auth_headers, app, fake_redis, tmp_path):
    dataset_id = _summary_dataset(app, tmp_path, ROWS)

    model_id, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)

    assert job.status == "completed", job.error_message
    for name in ("rouge1", "rouge2", "rougeL"):
        assert 0.0 <= job.metrics[name] <= 1.0
    assert job.metrics["rouge_rows"] >= 1
    bundle = joblib.load(get_model_path(app.config["MODEL_FOLDER"], model_id))
    assert bundle["task"] == "summarization"
    assert bundle["max_source_length"] == 128 and bundle["max_target_length"] == 32
    assert bundle["prefix"] == ""  # a BART stand-in takes the bare text; T5 gets "summarize: "


def test_prediction_returns_a_summary_per_text(client, auth_headers, app, fake_redis, tmp_path):
    dataset_id = _summary_dataset(app, tmp_path, ROWS)
    model_id, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)
    assert job.status == "completed", job.error_message

    texts = ["The court held the term void.", "The appeal was allowed."]
    result = process_prediction_task(
        {"prediction_id": "p", "model_id": model_id, "model_type": "t5_small", "texts": texts},
        app.config["MODEL_FOLDER"],
    )

    assert [p["text"] for p in result["predictions"]] == texts
    assert all(isinstance(p["prediction"], str) for p in result["predictions"])


def test_blank_summaries_are_not_trained_on_and_texts_are_read_verbatim(app, tmp_path):
    from legal_ml_core.annotation_queue import summarization_examples

    path = tmp_path / "mixed.csv"
    path.write_text("text,summary\nNA,null\n00123,1.50\nA text without a summary,\n,A summary without a text\n")

    examples = summarization_examples(str(path))

    assert examples == [("NA", "null"), ("00123", "1.50")]


def test_a_dataset_with_no_summaries_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = _summary_dataset(app, tmp_path, [("A text", "")])
    message = "Add some summaries first: this Dataset has no rows with both a text and a summary to learn from."

    fit = client.get(f"/api/models/available?dataset_id={dataset_id}", headers=auth_headers).get_json()
    created = _create(client, auth_headers, dataset_id)

    assert fit["models"]["t5_small"]["usable"] is False
    assert fit["models"]["t5_small"]["reason"] == message
    assert created.status_code == 400
    assert created.get_json()["error"] == message


def test_a_one_row_dataset_says_it_was_scored_on_its_training_row(client, auth_headers, app, fake_redis, tmp_path):
    dataset_id = _summary_dataset(app, tmp_path, ROWS[:1])

    _, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)

    assert job.status == "completed", job.error_message
    assert job.metrics["evaluated_on_training"] is True


def test_a_stop_during_the_rouge_generation_ends_the_run_as_cancelled(app, tmp_path, monkeypatch):
    from legal_ml_core.compute import trainer as trainer_module

    # Only the generation pass can notice the Stop here.
    monkeypatch.setattr(trainer_module.TrainingCancelCallback, "on_step_end", lambda self, *a, **k: None)

    def stop():
        return inspect.stack()[1].function == "generated_summaries"

    trainer = trainer_module.MLModelTrainer(redis_client=None, cancel_check=stop)
    model_dir = tmp_path / "tiny-seq2seq"
    model_dir.mkdir()
    parameters = {**PARAMETERS, "model_name": _tiny_seq2seq(model_dir), "model_id": "stop",
                  "output_dir": str(tmp_path / "out")}

    with pytest.raises(trainer_module.TrainingCancelled):
        trainer.train_seq2seq(ROWS, "t5_small", parameters)


def test_t5_small_is_listed_for_summarization(client, auth_headers):
    resp = client.get("/api/models/available?task_type=summarization", headers=auth_headers)

    assert resp.status_code == 200
    assert "t5_small" in resp.get_json()["models"]


def test_explanations_and_export_refuse_summary_models(client, auth_headers, app, fake_redis, tmp_path):
    from legal_ml_core.compute.export import process_export_task

    dataset_id = _summary_dataset(app, tmp_path, ROWS)
    model_id, job = _train(client, auth_headers, app, fake_redis, dataset_id, tmp_path)
    assert job.status == "completed", job.error_message

    single = client.post(f"/api/models/{model_id}/explain", json={"text": "A text"}, headers=auth_headers)
    batch = client.post(f"/api/models/{model_id}/explain-batch", json={"texts": ["A text"]}, headers=auth_headers)
    assert single.status_code == 400
    assert batch.status_code == 400

    with pytest.raises(ValueError, match="Only classification Models"):
        process_export_task(
            {"model_id": model_id, "model_type": "t5_small", "texts": ["A text"]},
            app.config["MODEL_FOLDER"],
        )
