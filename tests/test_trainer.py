import pandas as pd
import pytest

from legal_ml_core.compute.trainer import MLModelTrainer


@pytest.fixture
def classification_df():
    texts = [
        "the court ruled in favor of the plaintiff",
        "the defendant was found liable for damages",
        "judgment was entered against the appellant",
        "the parties reached a settlement agreement",
        "the tribunal dismissed the case entirely",
        "the appeal was rejected by the higher court",
        "the jury returned a verdict of guilty",
        "the contract was declared void by the judge",
    ] * 3
    labels = ["civil", "criminal"] * 12
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.mark.parametrize(
    "model_type",
    ["svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression"],
)
def test_train_classification_model_all_algorithms(fake_redis, classification_df, model_type):
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, labels, label_names = trainer.preprocess_data(
        classification_df, "classification", {}
    )

    parameters = {"model_id": "test-model-1"}
    metrics = trainer.train_classification_model(texts, labels, model_type, parameters)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    assert fake_redis.get("confusion_matrix_test-model-1") is not None


def _fitted_estimator(trainer):
    calibrated = getattr(trainer.model, "calibrated_classifiers_", None)
    return calibrated[0].estimator if calibrated else trainer.model


@pytest.mark.parametrize(
    "model_type", ["svm", "decision_tree", "random_forest", "logistic_regression"]
)
@pytest.mark.parametrize("chosen, expected", [("balanced", "balanced"), ("none", None)])
def test_class_weight_reaches_the_estimator(fake_redis, classification_df, model_type, chosen, expected):
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, labels, _ = trainer.preprocess_data(classification_df, "classification", {})

    trainer.train_classification_model(
        texts, labels, model_type, {"model_id": "weighted", "class_weight": chosen}
    )

    assert _fitted_estimator(trainer).class_weight == expected


@pytest.mark.parametrize("model_type", ["kmeans", "hierarchical", "dbscan"])
def test_train_clustering_model(fake_redis, classification_df, model_type):
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, _, _ = trainer.preprocess_data(classification_df, "clustering", {})

    parameters = {"model_id": "test-model-2", "n_clusters": 2}
    metrics = trainer.train_clustering_model(texts, model_type, parameters)

    assert "n_clusters" in metrics
    assert fake_redis.get("clustering_data_test-model-2") is not None


@pytest.mark.parametrize("model_type", ["lda", "nmf"])
def test_train_topic_modeling(fake_redis, classification_df, model_type):
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, _, _ = trainer.preprocess_data(classification_df, "topic_modeling", {})

    parameters = {"model_id": "test-model-3", "n_components": 2}
    metrics = trainer.train_topic_modeling(texts, model_type, parameters)

    assert metrics["n_topics"] == 2
    assert fake_redis.get("topic_data_test-model-3") is not None


def test_save_and_reload_model(fake_redis, classification_df, tmp_path):
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, labels, _ = trainer.preprocess_data(classification_df, "classification", {})
    trainer.train_classification_model(texts, labels, "naive_bayes", {"model_id": "m4"})

    model_path = tmp_path / "m4.pkl"
    trainer.save_model(str(model_path))
    assert model_path.exists()

    import joblib

    loaded = joblib.load(str(model_path))
    assert loaded["model"] is not None
    assert loaded["vectorizer"] is not None
    assert loaded["label_encoder"] is not None


@pytest.mark.parametrize("model_type", ["logistic_regression", "naive_bayes"])
def test_labels_with_a_single_row_still_train(fake_redis, classification_df, model_type):
    # A stratified split needs two rows of every label; with a label seen once
    # (common in imported data) the split is made without stratifying.
    rare = pd.DataFrame({"text": ["a one-off ruling on maritime salvage", "a lone tax matter"], "label": ["salvage", "tax"]})
    df = pd.concat([classification_df, rare], ignore_index=True)
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, labels, _ = trainer.preprocess_data(df, "classification", {})

    metrics = trainer.train_classification_model(texts, labels, model_type, {"model_id": "rare-1"})

    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.fixture
def integer_label_df(classification_df):
    """Hugging Face ClassLabel columns and plain 0/1/2 CSVs arrive as integers
    (BAIGLE #218)."""
    df = classification_df.copy()
    df["label"] = pd.Series([0, 1, 2] * 8, dtype="int64")
    return df


@pytest.mark.parametrize(
    "model_type",
    ["svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression"],
)
def test_integer_labels_train_and_read_as_text(fake_redis, integer_label_df, model_type):
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, labels, _ = trainer.preprocess_data(integer_label_df, "classification", {})

    metrics = trainer.train_classification_model(
        texts, labels, model_type, {"model_id": "int-labels"}
    )

    assert metrics["confusion_matrix_labels"] == ["0", "1", "2"]
    assert len(metrics["confusion_matrix"]) == 3
    assert trainer.label_encoder.classes_.tolist() == ["0", "1", "2"]


def test_csv_floats_and_saved_text_labels_are_one_class(fake_redis, classification_df):
    """A CSV column with a blank reads as floats (1.0); saved annotations are
    text ("1"). Both are the label 1."""
    df = classification_df.copy()
    df["label"] = pd.Series([0.0, 1.0] * 6 + ["0", "1"] * 6, dtype=object)
    trainer = MLModelTrainer(redis_client=fake_redis)

    trainer.preprocess_data(df, "classification", {})

    assert trainer.label_encoder.classes_.tolist() == ["0", "1"]


def test_true_false_labels_train_as_text(fake_redis, classification_df):
    df = classification_df.copy()
    df["label"] = pd.Series([True, False] * 12, dtype=bool)
    trainer = MLModelTrainer(redis_client=fake_redis)
    texts, labels, _ = trainer.preprocess_data(df, "classification", {})

    metrics = trainer.train_classification_model(texts, labels, "logistic_regression", {"model_id": "bool"})

    assert metrics["confusion_matrix_labels"] == ["False", "True"]


def test_rows_without_a_label_are_left_out(fake_redis, classification_df):
    """A direct caller's blank labels (NaN, "", whitespace) never become a class."""
    df = classification_df.copy()
    df["label"] = pd.Series(["a", "b", None, "", "   ", float("nan")] * 4, dtype=object)
    trainer = MLModelTrainer(redis_client=fake_redis)

    texts, labels, names = trainer.preprocess_data(df, "classification", {})

    assert sorted(names) == ["a", "b"]
    assert len(texts) == len(labels) == 8
    assert trainer.label_encoder.classes_.tolist() == ["a", "b"]


def test_a_file_with_no_labels_is_refused(fake_redis, classification_df):
    df = classification_df.copy()
    df["label"] = None
    with pytest.raises(ValueError, match="at least one labelled row"):
        MLModelTrainer(redis_client=fake_redis).preprocess_data(df, "classification", {})
