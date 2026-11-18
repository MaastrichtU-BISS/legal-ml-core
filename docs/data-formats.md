# Data formats

This page describes the files legal-ml-core reads and writes: datasets you
upload, and the model artifacts training produces. The HTTP routes are
described in [`openapi.yaml`](openapi.yaml).

## Dataset upload

`POST /api/datasets/upload` (multipart form) takes a `file` plus `name`,
`task_type`, and optionally `description`, `labels` and `annotation_mode`.

- **File types:** `.csv`, `.xlsx` or `.json` (anything pandas can read as a
  table with `read_csv`, `read_excel` or `read_json`).
- **Size:** at most 1,000 rows unless the host sets `LEGAL_ML_MAX_ROWS`. A larger file gets 400 `Dataset exceeds maximum allowed rows (<cap>)`.
- **Header row:** required. Required column names are checked
  case-insensitively (`Text` counts as `text`).
- **Storage:** the file is saved unchanged as `<uuid>_<original name>` in the
  host's `UPLOAD_FOLDER`. The `Dataset` row records its path, row count, and
  per-column dtype, null count and unique count.
- **`annotation_mode=true`:** the label column becomes optional, so you can
  upload unlabelled text and label it in the annotation queue.
- **`labels`:** the label set for annotation, comma-separated
  (`Obligation,Permission`) or as a JSON array string (`["Obligation","Permission"]`).

### Columns per task type

The catalogue is served by `GET /api/datasets/task-types` (source:
`TASK_TYPES` in `src/legal_ml_core/routes/datasets.py`).

| `task_type` | Required columns | Label column holds | Can be trained |
|---|---|---|---|
| `classification` | `text`, `label` | one category per passage (e.g. Obligation) | yes |
| `document_classification` | `text`, `label` | one category per whole document | yes |
| `judgment_prediction` | `text`, `judgment` | the case outcome (e.g. guilty / not_guilty) | yes |
| `topic_modeling` | `text` | nothing (unsupervised) | yes |
| `clustering` | `text` | nothing (unsupervised) | yes |
| `ner` | `text`, `entities` | entity spans as JSON, or the entity types (e.g. `PERSON, ORG`) | yes |
| `named_entity_recognition` | `text`, `entities` | same as `ner` (alias) | yes |
| `summarization` | `text`, `summary` | the target summary | not yet |

"Not yet" task types can be uploaded, previewed, explored (EDA) and annotated,
but training them returns 400.

An NER `entities` cell may hold the annotated spans as a JSON list, with
character offsets into that row's `text` (`end` exclusive):

```csv
text,entities
"John Smith sued Acme Corp.","[{""text"": ""John Smith"", ""start"": 0, ""end"": 10, ""type"": ""PERSON""}, {""text"": ""Acme Corp."", ""start"": 16, ""end"": 26, ""type"": ""ORG""}]"
```

Every span must fit the row's text, be in order, and not overlap; `[]` is a
valid row with no entities. Such rows start labelled in the annotation queue and
train `legalbert_ner` or `distilbert_ner`. A cell naming types instead (e.g.
`PERSON, ORG`) is kept as an annotation hint and the row waits for its spans.

An NER file is read exactly as written (no type guessing): `NA`, `null`,
`00123` or `1.50` stay text, so the spans' offsets still point into it. A row
whose cell looks like spans but does not fit its text stays pending, and the
worker logs the dataset and row. Training needs at least one annotated,
non-holdout row with an entity (rows annotated `[]` alone teach nothing). Texts
longer than the model's `max_length` are trained and predicted in overlapping
windows, so entities past the first window are kept; prediction labels whole
words, taking each word's label from its first sub-token.

Example (`classification`):

```csv
text,label
"Member States shall adopt these laws",Obligation
"The Commission may request information",Permission
"Operators shall not export the waste",Prohibition
```

### Requirements for training

Upload validation checks column *names*, but the trainer reads columns *by
position* and reads the file as CSV. For a dataset you want to train on:

1. Upload it as **CSV** (XLSX and JSON pass upload and EDA, but training reads
   the stored file with `pandas.read_csv`).
2. Put the **text in the first column**, and for supervised tasks the
   **label (`label`, `judgment`, `summary` or `entities`) in the second column**.
   Extra columns after these are ignored. NER training reads the saved spans on
   the Dataset's annotation tasks, so its raw text keeps its offsets.

## Trained models

### Where they are stored

The worker saves artifacts under the `model_storage_path` given to
`run_ml_worker` (the API process receives the same root as
`init_ml_feature(..., model_storage_path=...)`, stored in
`LEGAL_ML_MODEL_STORAGE_PATH`). With the default `LocalStorageBackend`
(`src/legal_ml_core/storage.py`):

| Artifact | Path |
|---|---|
| Model bundle (every model type) | `<model_storage_path>/<model_id>.pkl` |
| Transformer working directory (LegalBERT/RoBERTa) | `<model_storage_path>/transformer_<model_id>/` |

A host with object storage can replace the layout by setting
`LEGAL_ML_STORAGE_BACKEND` to an object with `get_model_path`,
`get_transformer_output_dir` and `get_dataset_path`.

The bundle is written only when training completes. If the job is stopped or
fails, any file already written is removed.

### Bundle format

`<model_id>.pkl` is a [joblib](https://joblib.readthedocs.io/) pickle of a
dict (`Trainer.save_model` in `src/legal_ml_core/compute/trainer.py`):

| Key | Contents |
|---|---|
| `task` | `"ner"` for an entity model, else `None` |
| `preprocessor` | the fitted `TextPreprocessor` (cleaning options used at training); `None` for entity models, which keep raw text |
| `vectorizer` | fitted scikit-learn vectorizer, or `None` for transformers |
| `model` | the scikit-learn estimator, or the fine-tuned HuggingFace model |
| `label_encoder` | fitted `LabelEncoder` for supervised tasks, else `None` |
| `tokenizer` | HuggingFace tokenizer for transformers and entity models, else `None` |
| `label_list` | entity models only: the token labels, `O` first |
| `max_length` | entity models only: the longest text the tokenizer reads |

Loading it needs Python with this package and its pinned dependencies
installed (the preprocessor class lives in `legal_ml_core.compute`):

```python
import joblib
bundle = joblib.load("model.pkl")
X = bundle["vectorizer"].transform([bundle["preprocessor"].clean_text(t) for t in texts])
labels = bundle["label_encoder"].inverse_transform(bundle["model"].predict(X))
```

Only load bundles you trust: unpickling runs code.

### Metrics and parameters

Hyperparameters are stored as JSON in `Model.parameters`. Evaluation results
(accuracy, precision, recall, F1, confusion matrix and its labels,
classification report, or clustering/topic outputs) are stored as JSON in
`Model.metrics` and `TrainingJob.metrics`. Per-step curves are rows of
`TrainingMetricPoint` (`GET /api/training/<job_id>/metrics`). Each point has a
`group` (`train`, `eval`, `system` or `stage`); fine-tuning points also have
`training_step`, the trainer's own step, while `step` is only the polling cursor.
Learning-curve points (`learning_curve_*`, with `monitoring.learning_curve` on)
carry the training-set size as `training_step`.

### Export

`GET /api/models/<model_id>/download` returns a link and filename for a
completed model; `GET /api/models/<model_id>/file` sends the bundle as an
attachment named `<model name>_<model_type>.pkl`.
