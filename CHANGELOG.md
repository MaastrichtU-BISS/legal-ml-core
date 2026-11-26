# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

## [0.22.1] - 2026-11-26

### Fixed

- Training a Dataset whose label column holds integers (a Hugging Face
  `ClassLabel` without names, or a CSV of 0/1/2) no longer fails with "object
  of type 'int' has no len()" (BAIGLE #218). The trainer reads every label as
  text, as the annotation queue does: 1, 1.0 (a CSV column with a blank) and a
  saved "1" are one class. Metrics, the confusion matrix, the saved label
  encoder and predictions show the labels as text. Datasets with text labels
  are unchanged.
- A failed training job's worker log carries its traceback, not just the
  message.

## [0.22.0] - 2026-11-26

### Added

- `POST /api/datasets/<id>/practice-copy` makes an unlabelled practice copy of
  a labelled single-label Dataset (BAIGLE #197), so a student can practise LLM
  labelling without losing their labels. The copy is owned by the same user, in
  the same project, named "<original> (practice)", with the same text and label
  list and no labels: its file keeps only the text and a blank label column,
  and every row is a pending task. The original's current prompt template and
  definitions become the copy's prompt version 1. The original is unchanged.
  Other task types, and a Dataset with no labels yet, are refused with a 400.
- `Dataset.origin_dataset_id` records the Dataset a practice copy was made
  from, and `GET /api/datasets/<id>` returns it (null for an original).
  Migration `lmc_0009_dataset_origin` adds the nullable column; hosts that
  chain their own migrations onto the package need a merge revision with it.

## [0.21.0] - 2026-11-25

### Changed

- The worked examples in an LLM suggestion prompt are balanced across labels
  (BAIGLE #195): every label with a confirmed item gets one example before any
  label gets a second, in the order the Dataset names its labels, up to the
  existing 5. They are still drawn only from confirmed (`human`,
  `llm_confirmed`), non-held-out items, earliest items first, and are listed in
  item order. Before, the first 5 confirmed items were sent, so a label
  labelled later could be missing from every prompt.

## [0.20.0] - 2026-11-25

### Added

- `PUT /api/datasets/<id>/label-set` names a single-label Dataset's labels
  after upload (BAIGLE #193), so a Dataset uploaded without a label list, or
  imported from a library, can use LLM labelling. The body is exactly
  `{labels: [string]}`: names are trimmed, 2 to 50 of them, 1 to 100
  characters each, none equal ignoring case. Until an item is labelled the
  list may be replaced freely; after that it must keep every label an
  annotated task has (from any source) and may only add labels or change
  unused ones. Only `classification`, `document_classification` and
  `judgment_prediction` Datasets accept it. Returns `{dataset}` like `GET`;
  refusals are 400 with a plain `error`, and someone else's Dataset is 404.
  The LLM prompt and active-learning state are unchanged.
- Bulk accept skips pending LLM suggestions for a label the Dataset no
  longer names, so a removed label is never written back as a label.
- `PATCH /api/llm/key` changes the saved key's model without the key (#27),
  so a host can offer "Change model" after listing models with
  `{use_saved_key: true}`. The body is exactly `{model}`, a non-empty string of
  at most 255 characters after trimming, with no NUL and no repeated member,
  sent as `application/json`; anything else is 400 `invalid_request`. No saved
  key, or one removed while the change runs, is 404 `no_saved_key`. Returns
  `{base_url, model}`, like `PUT`. The encrypted key and `base_url` are left
  alone, and the key is never decrypted. `PUT` and `DELETE` are unchanged.

## [0.19.0] - 2026-11-24

### Added

- `POST /api/llm/models` lists the chat models an OpenAI-compatible provider
  offers (#25), for checking a key before saving it and picking a model from a
  list. The body is required: `{base_url, api_key}`, a key that is never
  stored, or `{use_saved_key: true}`. Returns `{models: [{id, name?}]}` sorted
  by id; `name` is left out when it equals the id.
  - Which models: on OpenRouter, only models whose `output_modalities` (or
    `modality` string) say they output text; a model with no such metadata is
    left out. Elsewhere, every id that doesn't name embed, whisper, tts (as a
    whole token), dall-e, image, audio, moderation, realtime or transcribe.
    Both `{data: [...]}` and a bare top-level list (Together) are read; blank
    ids are dropped, and so is any model whose id contains the key, or the
    name of one whose name does.
  - Checking the key: OpenRouter lists its models to any key, so
    `GET {base_url}/key` is asked first and a 401/403 there is `key_rejected`.
    `api.anthropic.com` gets the key as `Authorization: Bearer` only (no
    `x-api-key`), with `anthropic-version: 2023-06-01` and `?limit=1000`.
  - Inputs: the key must be printable ASCII with no whitespace, at most 512
    characters (`invalid_request`, "The API key has characters an API key
    can't contain."); a saved key that isn't is `saved_key_unreadable`. The
    base URL passes `validate_byok_url` on every call, and one that is
    malformed or has a query or fragment is `base_url_refused`.
  - Transport: urllib3 on a single connection, not requests. No redirect is
    followed or read; `Accept-Encoding: identity` is sent and any other
    `Content-Encoding` is refused unread, so nothing is ever decompressed;
    bodies are read in bounded amounts against a 2 MB cap. One wall-clock
    budget of `max(1, min(LLM_TIMEOUT_SECONDS, 15))` seconds covers each
    connection attempt (at most 5 s each, tried address by address over the
    address families urllib3 itself would use, so an IPv6 address the host
    can't open falls through to IPv4), the TLS
    handshake, the headers and the body, enforced by a watchdog that cuts the
    socket, so a provider trickling bytes can't hold the request open. The
    DNS lookup is not covered: it is left to the system resolver, and bounding
    it (with DNS pinning) is a tracked follow-up.
  - urllib3's "Failed to parse headers" log, which can carry header text a
    provider echoed back, is dropped for the listing's own thread while it
    runs; other users of urllib3 log as before. A crafted `Content-Length`
    (`²`, thousands of digits) is refused or ignored, never a 500. A base URL
    host with a trailing dot (`openrouter.ai.`) gets the same provider
    handling as without it.
  - Refusals carry a `code`: `key_rejected` (401, 403, or a 400 whose first
    8 KB say "API key", as Gemini answers a bad key), `provider_unreachable`
    (502: network errors, the deadline, 408, 5xx, and 429 with "The provider is
    limiting requests. Try again in a moment."), `listing_unsupported` (any
    other status, a redirect, a compressed, oversized or unreadable body, or
    a non-empty list without one model id; an empty list, or one the chat
    filter empties, is still a 200 with no models),
    `no_saved_key`, `saved_key_unreadable`, `base_url_refused` and
    `invalid_request`. A refusal never carries the library exception (which
    can hold the request headers) as its context. No rate limit: the host owns
    that policy.
- `urllib3>=2,<3` is now a direct dependency (it already came with requests).

## [0.18.0] - 2026-11-23

### Fixed

- `GET /api/llm/status` answers for a shared-key user even while the usage
  meter is down (#166): the response still says `mode: "shared"` with
  `remaining_tokens` unknown, and new fields `shared_key_available: false` and
  `shared_key_unavailable_reason` (the message that was previously the whole
  error) let the panel offer the user's own API key instead of a failure.
  Saving an own key still switches the account to BYOK, and everything that
  would spend the shared key — suggestions, prompt checks, annotation jobs —
  keeps failing closed until the meter answers again.

## [0.17.0] - 2026-11-18

### Added

- Summarization Datasets train end to end: `POST /api/models/` accepts
  `t5_small` ("T5 small (summaries)") in a new `seq2seq` catalogue category,
  and `summarization` now lists it. It trains on the Dataset's `text` and
  `summary` columns, found by name and read exactly as written (never cleaned
  or lowercased); rows with a blank text or summary are skipped. Runs report
  `rouge1`, `rouge2` and `rougeL` (0–1, `rouge-score`, words in any alphabet with
  their combining marks, no stemming) on greedy summaries written for up to 200 evaluation rows
  (`rouge_rows`); a Stop during that generation ends
  it at the next batch, and a one-row Dataset's metrics say
  `evaluated_on_training`.
- Prediction for `t5_small` returns the written summary as `prediction`
  (beam search, a few texts at a time; a blank text gets `""`); the predict route waits up to 300 s
  for it.
- `rouge-score` and `sentencepiece` in the `compute` extra.

### Changed

- A summarization Dataset with no row holding both a text and a summary is
  refused with "Add some summaries first: this Dataset has no rows with both a
  text and a summary to learn from." `UNSUPPORTED_TASK_TYPES` is now empty.
- Explanations are refused with a plain 400 for `t5_small`, and publishing
  refuses it.

## [0.16.0] - 2026-11-18

### Added

- NER Datasets train end to end: `POST /api/models/` accepts `legalbert_ner`
  ("LegalBERT (entities)") and `distilbert_ner` ("DistilBERT (entities)") in a
  new `token_classification` catalogue category, and `ner` /
  `named_entity_recognition` now list them. Runs report `f1_score`,
  `precision`, `recall`, `classification_report` and `entity_types`, scored on
  spans with `seqeval`.
- Prediction for entity models returns `entities`: `{start, end, type, text,
  score}` spans with character offsets into the input text, and `prediction` as
  the distinct entity types, first seen first (`""` when none).
- An imported NER row whose `entities` cell holds a valid JSON span list starts
  annotated (`label_source: file`), so imported spans train without re-labelling;
  `[]` is a valid row with no entities. Cells naming types (`PERSON, ORG`) keep
  their annotation hint and stay pending. An import trains straight away,
  without its rows being opened in a project's queue: creating a Model,
  starting a run and the worker create an NER Dataset's missing tasks (other
  task types are unchanged), while asking which
  models a Dataset can train only reads (its file, when its rows are not
  queued yet). A rejected cell that looks like spans is logged with the
  dataset and row.
- NER files are read exactly as written (`NA`, `null`, `00123` stay text), so
  span offsets keep pointing into the text; a JSON file's `entities` may be a
  real JSON array as well as JSON text.
- Texts longer than `max_length` are trained and predicted in overlapping
  windows instead of being cut, and predicted a few windows at a time;
  prediction labels whole words from each word's first sub-token, so a word is
  never split into two entities. A run's final scores are per row, each word
  labelled as prediction would, so overlapping windows are not counted twice;
  the entity scores logged after each evaluation (`eval_f1_score`,
  `eval_precision`, `eval_recall`) are per row the same way, and a Stop asked
  for during that scoring pass ends it at the next batch of windows.
- `seqeval` in the `compute` extra.

### Changed

- An NER Dataset with no annotated, non-holdout row holding an entity (rows
  annotated `[]` alone do not count) is refused with "Annotate some rows first:
  this Dataset has no labelled entities to learn from." NER text is never
  cleaned for training, so span offsets stay inside it. An evaluation split
  with no entities on either side scores 0 with an empty report instead of
  failing, and a one-row Dataset's metrics say `evaluated_on_training`.
- Explanations are refused with a plain 400 for entity models, and publishing
  still refuses them.

## [0.15.0] - 2026-11-17

### Added

- `GET /api/datasets/limits` returns `{"max_rows": n}`: the row cap a new
  dataset meets (the host's `LEGAL_ML_MAX_ROWS`, else 1000), read on each call
  and public like the task types, so upload copy can show the cap an admin set
  (reducing_digital_divide #118).

## [0.14.1] - 2026-11-17

### Added

- `GET /api/llm/status` returns `shared_key_configured`: whether this deployment
  offers a shared key at all, for any account. Nothing about the key itself is
  returned. `shared_key_allowed` keeps its meaning, so a client can now tell "no
  shared key here" from "this account may not use it" (reducing_digital_divide
  #155).

## [0.14.0] - 2026-11-17

### Added

- Summarization, folded in from `legal-summarizer-service`: `POST /api/summarize`
  queues a summary on `summarization_queue`, `GET /api/summarize/<request_id>`
  returns its status and result to its owner only, and `GET /api/summarize/models`
  lists the models. Extractive (sumy: LexRank, LSA, TextRank, Luhn, Edmundson)
  and abstractive (Hugging Face, limited to `LEGAL_ML_SUMMARIZATION_HF_MODELS`).
  Clients poll; there is no SSE stream. A 400 names the `field` to fix.
- `sumy` in the `compute` extra; the worker downloads NLTK punkt on start.

### Changed from legal-summarizer-service

- An unknown model is refused with 400 instead of falling back to LexRank.
- Edmundson works: it is given legal cue words (it always failed before).
- Input is capped at `LEGAL_ML_SUMMARIZATION_MAX_INPUT_CHARS` (500 000).

## [0.13.0] - 2026-11-13

### Changed

- Every row with text is an annotation task, labelled in the file or not
  (BAIGLE #147). A row the file labels starts `annotated` with
  `label_source: file`, trains as before, and can be opened in the kit and
  changed; a change is a human label. Before, only blank rows had tasks, so a
  fully labelled Dataset had nothing to annotate and file labels could not be
  corrected.
- NER files name entity types, not spans, so every NER row is pending; the
  types come back as `hint` from `GET /api/annotation/tasks/<id>/assignment`.
- Labels are read from tasks only: active learning, the holdout and training
  count each row once. A Dataset queued before gets its file-labelled tasks on
  its next read (once; marked `all_rows_queued`), and a loop started before
  stops adding its stored file counts on top.

### Added

- Migration `0008`: `annotation_tasks.file_label`, the file's own value for the
  row, kept after a person changes the label. In `to_dict` and the assignment.
- `GET /api/annotation/projects/<id>/tasks` takes `label`, `include_holdout`,
  `limit` and `offset`, in row order, and returns `total`: a review of existing
  labels, a page at a time.

## [0.12.1] - 2026-11-13

### Fixed

- Saving a kit Assignment (`PUT /api/annotation/tasks/<id>/assignment`) and
  importing a kit file failed with a 500 on Postgres: the task id, a `uuid.UUID`
  there, went into the JSON payload as is. Ids are now stored as text
  (`annotation_format.stored_payload`), and an Assignment built from a label
  carries its id and annotator as text too.
- A Dataset imported from a kit file could not train ("empty vocabulary"): the
  import stored its rows as JSON, but training reads the file as CSV. The import
  now writes CSV.

## [0.12.0] - 2026-11-13

### Added

- A task's annotation is a vue-legal-annotation-kit Assignment (BAIGLE #137).
  `GET`/`PUT /api/annotation/tasks/<id>/assignment` open and save one; it is
  stored as sent in `annotation_payload`, and `annotation_label` is derived from
  it in `annotation_format`: one document tag from the Dataset's labels for the
  single-label types, the `[{text, start, end, type}]` entity list for NER. A
  broken rule is a 400 that says which. A task saved before opens with an
  Assignment built from its label.
- Annotations export as kit TaskData, `GET /api/datasets/<id>/annotations/export`,
  and a kit file imports as a Dataset, `POST /api/datasets/import/legal-annotation-kit`
  (BAIGLE #138). Relations, confidence and metadata survive a round trip.

### Changed

- Migration `0007`: `annotation_tasks.annotation_payload` (JSON) is added and
  `annotation_label` widens to Text. Hosts run it before deploying. Downgrade cuts
  labels over 1000 characters to fit.
- The plain `/save` and LLM confirm replace a task's Assignment (it is rebuilt from
  the label), reset clears it, and correcting a typed label renames it inside it.

## [0.11.2] - 2026-11-13

### Fixed

- A Model made in a Project now brings its Dataset into that Project when the
  Dataset is in none, so the Project page lists the Dataset with its Model.
  A Dataset already in another Project stays there.
- Training failed with sklearn's "least populated classes ... only 1 member"
  when a label had a single row. The train/test split now stratifies only when
  it can, and otherwise splits at random.

## [0.11.1] - 2026-11-13

### Fixed

- Creating, training or listing models for a Dataset whose `columns` are bare
  names (no counts) failed with a 500. Such a Dataset now fits by task type alone.

## [0.11.0] - 2026-11-13

### Added

- `GET /api/models/available?dataset_id=<id>`: every model, with `usable` and a
  `reason` a student can act on, and `advice` when none fits. Decided from the
  task type and the label counts stored with the Dataset (no file read).
- `model_fit(dataset)` and `model_fit_refusal(model_type, dataset)`.

### Changed

- Creating or training a classifier or transformer on a Dataset whose labels are
  not categories (fewer than 2 labelled rows per label on average, e.g. answers
  uploaded as labels) answers 400 with the reason, instead of training a model
  that cannot learn.

## [0.10.0] - 2026-11-12

### Added

- `create_dataset(owner_id, file_path, name, description, task_type,
  annotation_mode, labels)`, the one path from a file to a Dataset. It raises
  `DatasetInvalid` with a student-readable message for every refusal. The upload
  route now wraps it (BAIGLE #110).
- `LEGAL_ML_MAX_ROWS` host config: the row cap, as a whole number or a function
  returning one (read on every upload). Unset, it stays 1,000. A cap of 0 or less
  raises `ValueError`.

### Changed

- **Breaking:** an unexpected error during upload is no longer caught and returned
  as 500 `{"error": "<exception text>"}`. It reaches the host's error handler, and
  the saved file is still removed. Refusals are unchanged (400, same messages).

## [0.9.0] - 2026-11-10

### Changed

- **Breaking:** a dataset upload is limited to 1,000 rows, down from 10,000
  (`MAX_ROWS` in `routes/datasets.py`). A larger file gets 400
  `Dataset exceeds maximum allowed rows (1000)` and nothing is stored. Datasets
  already stored are not affected. Tests now cover the boundary.

## [0.8.0] - 2026-11-06

### Added

- An LLM-suggested first active learning batch: `seed_source: "llm"` on
  `POST /api/active-learning/datasets/<id>/start`, with `confirm_estimate: true`.
  The LLM labels three batches' worth of the pool on the worker, never the
  holdout, and the first batch takes each suggested label in turn, rarest first,
  so a rare label is in it. `POST .../seed-estimate` prices it beforehand without
  creating anything, and the loop state's `seed` reports the job.
  `seed_sources.llm` in the settings is now `true` (BAIGLE #67).
- `GET /api/llm/status` reports `max_sync_tasks`, the most items one
  suggestions request may ask for, so a UI need not assume the default.
- Exporting a classification Model for publishing (BAIGLE #69, ADR 0006):
  `legal_ml_core.model_export.request_model_export` asks the worker, on the
  prediction queue, to write skops (scikit-learn) or safetensors plus tokenizer
  (transformers) files with `preprocessing.json` and a `predict.py` reference
  script, then checks that script's predictions against the platform's on the
  given texts. `skops` joins the `compute` extra.

## [0.7.0] - 2026-11-05

### Fixed

- `test_size` in a training start request sets the share of rows held out for
  testing. It was ignored, and every run held out 20%. A value that is not more
  than 0 and less than 1 is refused with 400 before anything is queued.
- Logistic regression is built from `l1_ratio` and `C` instead of `penalty`, which
  scikit-learn 1.10 removes: `l1` is a ratio of 1, `l2` of 0, and `none` an
  infinite `C`. The catalogue's settings are unchanged.
- An upload over the host's `MAX_CONTENT_LENGTH` gets a 413 saying the file is
  too large, with the limit, whether or not the request declares its size. It got
  a 500.

### Changed

- Needs scikit-learn 1.8 or later, the first where `l1_ratio` alone picks the
  penalty. Still capped below 1.10 until the suite runs on it.

## [0.6.0] - 2026-11-05

### Added

- With `monitoring.learning_curve` on, a classical Text Classification run also
  stores a learning curve: `learning_curve_train_accuracy` (group `train`) and
  `learning_curve_eval_accuracy` (group `eval`) at five growing training-set
  sizes, the size as `training_step`. Logistic regression also stores `loss` and
  `eval_loss` by iteration, the iteration as `training_step`. It makes training
  slower, so it stays off by default, and a Stop is honoured between fits.
  Each model type in `GET /api/models/available` says whether it can, as
  `supports_learning_curve`.

- Each run can set its own monitoring under `parameters.monitoring`:
  `log_every_n_steps`, `eval_strategy` (`epoch` or `steps`) with `eval_steps`, and
  `include` / `exclude` name globs for which metric points are stored (stage markers
  always are), and `flush_every_points` (50) / `flush_every_seconds` (2.0) for how
  often the worker writes metric points in a batch, plus `learning_curve` and
  `system_metrics` (both off by default, described below).
  The host sets the defaults in `LEGAL_ML_MONITORING`, and a bad default stops
  `init_ml_feature`. A bad setting is refused with a 400 whose `field` is
  `monitoring.<setting>`. Fine-tuning used to log every 10 steps and evaluate each
  epoch; those are now the defaults.
- A run's `parameters` record the monitoring it used, as `monitoring`, with the
  defaults filled in. The queue message carries it as `monitoring`. Group names
  survive a retrain that changed only the monitoring.
- With `monitoring.system_metrics` on, a run records resource and throughput
  readings as `system` points: `memory_mb`, `gpu_memory_mb` and `gpu_use_percent`
  (only when CUDA is available), `samples_per_second` and, when fine-tuning,
  `seconds_remaining`. Fine-tuning takes them every `log_every_n_steps`; a
  classical run takes them at each stage, and times its speed over the fit of
  the model it keeps, in training rows a second, so a learning curve's extra fits
  do not slow it. Off by default. No new dependency:
  memory comes from psutil when it is installed, else from the standard library.

- Each metric point says what it means: `x_kind` is what its `training_step`
  counts (`step`, `iteration` or `training_examples`), and `unit` is a system
  point's unit (`MB`, `%`, `samples/s` or `s`, null for other points). Both are
  worked out from the stored point, so older points have them too and no
  migration is needed.

- Each training job records `parameters`, the Model's settings when the run
  started, so retraining no longer loses what an earlier run used. Jobs from
  before this release have `parameters: null`.
- `GET /api/models/<id>/runs` lists a Model's runs newest first, owner-scoped,
  with `parameters`, the headline `scores`, `status`, `error_message` and dates.
  It lists at most 50, and `has_more` says whether older runs exist.
- Migration `lmc_0005_training_job_parameters` adds `training_jobs.parameters`.
  Hosts with their own Alembic graph need a merge revision.
- Training metric points carry a `group` (`train`, `eval`, `system` or `stage`) and,
  for fine-tuning, `training_step`, the trainer's own step. `step` stays the
  polling cursor. Points stored before this release get a group from their name:
  `eval_*` is eval, the run's milestones (`training_started`, `model_saved`, ...)
  are stage, and the rest are train.
- Migration `lmc_0006_metric_point_group` adds
  `training_metric_points.metric_group` and `training_step`, both nullable. Hosts
  with their own Alembic graph need a merge revision.
- EDA for Text Classification adds `keyword_stats` beside `keywords_by_class`: each
  distinctive term with `in_class` (texts in the class that contain it), `overall`
  (texts that contain it) and `lift`. `keywords_by_class` keeps its shape.
- `GET /api/datasets/<id>/rows` takes `term` and `label` to show only the texts behind
  a distinctive term. `total_rows` and `has_more` then count the filtered rows, and the
  count matches `in_class`.

### Changed

- Fine-tuning no longer stores the trainer's end-of-run summary (`total_flos`,
  `train_runtime`, `train_loss`, the `*_per_second` rates) or `epoch` as metric
  points, so they no longer flatten the curves. Hosts that read those names from
  `GET /api/training/<job_id>/metrics` will not find them in new runs.
- The worker writes training metric points in batches, every 50 points or once the
  oldest has waited 2 seconds, instead of one commit per point. Stage markers are
  written at once, and every waiting point is written when a run completes, fails
  or is stopped. Polling `GET /api/training/<job_id>/metrics` can see a point up to
  2 seconds later than before, and `step` still has no gaps.
- Distinctive terms that tie on lift are ordered by how many texts in the class contain
  them, then alphabetically, so the order no longer changes between requests.
- `PUT /api/models/<id>` refuses a change of `parameters` with a 409 while the
  Model has a pending or running job. A rename, or the same parameters, still
  goes through.
- A successful retrain keeps the group names saved with `PUT
  /api/models/<id>/labels` (`metrics.display_labels`) only when the run used the
  same parameters as the run before it. Before, every retrain dropped them. A run
  from before `parameters` were recorded counts as different, so its names go.
- Run `parameters` hold the catalogue defaults as well as the Model's choices, and
  runs and the training guard compare parameters that way. A Model saved with some
  settings, then saved again by a form that sends them all, is not a change.

- **Breaking for hosts:** the worker's training stack (transformers with torch,
  accelerate, spaCy, NLTK, LIME and SHAP) moved to the `[compute]` extra. Install
  `legal-ml-core[compute]` wherever `run_ml_worker` runs. The API process needs
  only `legal-ml-core`, so an API image no longer carries torch. With PyPI's
  CUDA build, that was several GB.
- `run_ml_worker` is imported on first use, so `import legal_ml_core` and
  `init_ml_feature` no longer load the training stack.

## [0.5.2] - 2026-10-31

### Added

- A 400 refusing a Model's parameters carries `field`, the key of the parameter to
  change, beside `error`. A rule across two parameters names the one it limits, such
  as `solver`. Refusals tied to no one parameter have no `field`.
- Catalogue parameters have a `label`, the name the wizard shows.

### Changed

- Refusals use that label, so "Inverse regularisation strength must be…" now reads
  "Regularisation (C) must be…", and limits read 0.000001, not 1e-06. Anything matching
  the old wording needs updating.
- `scikit-learn` is capped below 1.10, which removes the `penalty` the trainer passes.

## [0.5.1] - 2026-10-31

### Fixed

- **Every earlier release ignored the parameters chosen for a Model.** The worker
  handed the trainer the queue message, where the choices sit one level down under
  `parameters`, so classifiers, clustering, topic models and transformers all
  trained on defaults while `Model.parameters` showed the choice. This predates
  v0.2.0 and affects every release up to and including v0.5.0. Models trained so far
  used defaults and keep them until retrained: retraining applies their saved
  choices, so scores, cluster counts and training time may change.
- A balanced model keeps its benefit. Sigmoid calibration refitted its scores to the
  unweighted label frequencies and undid it, so `class_weight: balanced` now skips
  calibration and reports `probabilities_calibrated: false`.
- `GET /api/training/batch-prediction/<id>` answers `status: pending` from the moment
  a batch is queued, not 404 until the result arrives.
- `penalty: none`, the `newton-cg` and `sag` solvers and `cosine` hierarchical distance
  train, which the wizard always offered and the catalogue refused. `cosine` keeps a
  document with no usable words, which scikit-learn's own cosine option refuses.
- `max_depth: null`, the wizard's default for unlimited depth, is accepted.
- `penalty: elasticnet` mixes both penalties, through a new `l1_ratio` parameter
  (default 0.5). scikit-learn 1.8 and later otherwise fit it as plain l2.
- An NMF `alpha` strong enough to remove every topic fails with a reason.

### Changed

- Parameters that cannot train are refused with a 400 when a Model is created, when
  it is edited and when training starts, including Models already saved with them:
  `l1` without `liblinear` or `saga`, `elasticnet` without `saga`, and `ward` linkage
  without `euclidean` or `l2`.
- `lex_glue` is no longer a LegalBERT variant: no such model exists to download.
- Numbers are checked against their catalogue range on create, edit and start, in
  words: "Number of trees must be between 1 and 1000." Saved Models outside a range
  must be edited before they retrain. Some ranges widen to what the wizard allowed:
  `C` from 0.001, `degree` from 1, `n_estimators` from 1, `eps` from 0.01, `alpha`
  (Naive Bayes) from 0.01, K-Means `max_iter` from 10, `batch_size` from 4.
- `liblinear` is refused on a Dataset of more than two labels, read from the file
  only when `liblinear` is chosen. If the file cannot be read then, training fails
  with the same words. `n_clusters` is refused above the Dataset's rows.
- `GET /api/models/available` gives each model's `rules`, so a client can apply the
  dependent options, such as the solvers a penalty allows, without its own copy.
- NMF `l1_ratio` is no longer offered: an offered mix could remove every topic and
  fail in training. Pure L2 cannot. A saved value still applies. NMF `random_state`
  is offered instead.
- UK spelling across the model catalogue, including its descriptions.

## [0.5.0] - 2026-10-30

### Added

- `class_weight` parameter (`none` or `balanced`) for SVM, decision tree, random
  forest and logistic regression, in the trainer and the model catalogue.

### Fixed

- `/api/training/batch-predict` is served: the worker now consumes
  `batch_prediction_queue`, through `run_ml_worker(batch_prediction_queue=...)`.
  Before this, tasks stayed queued and the polling route answered 404.

### Security

- Requires PyJWT 2.15, whose JWKS client limits refreshes, so tokens with unknown
  key ids cannot force a fetch each. A test pins it.

## [0.4.0] - 2026-10-30

### Added

- Active learning for single-label text classification, under
  `/api/active-learning`: a holdout drawn once per dataset, a random seed batch,
  five query strategies on `predict_proba` output with NumPy and scikit-learn, a
  minimum per label up to a cap on labels, and a user-triggered retrain with a
  prompt. The worker rescores the pool on a new `scoring_queue`.
- The holdout draw is stratified on labels that already exist, with unlabelled
  rows random, and `holdout.label_counts` reports it by class.
- Label cap: a class short of its minimum holds training back only until
  `LEGAL_ML_AL_CAP_BATCHES` batches of labels. Past that the state names the short
  classes in `classes_short`, `classes_left_out` and `classes_unseen`. A class with
  one example is left out of training, and training still needs two classes with
  2 examples and enough rows for the stratified split.
- Migrations `lmc_0003_active_learning` (`al_configs`, `al_rounds`, and
  `annotation_tasks.score` and `is_holdout`) and `lmc_0004_holdout_file_counts`.
  Hosts with their own Alembic graph need a merge revision.
- `metrics.verification`: a model trained on a dataset with a loop is graded on
  its holdout, or reads `not_independently_verified`.
- Host config: `LEGAL_ML_AL_STRATEGIES`, `LEGAL_ML_AL_DEFAULT_STRATEGY`,
  `LEGAL_ML_AL_BATCH_SIZE`, `LEGAL_ML_AL_MIN_PER_CLASS`, `LEGAL_ML_AL_CAP_BATCHES`,
  `LEGAL_ML_AL_HOLDOUT_PERCENT`, `LEGAL_ML_AL_SCORING_STALE_SECONDS`,
  `LEGAL_ML_SCORING_QUEUE`.
- `PUT /api/active-learning/datasets/<id>/labels` corrects a typed label on every
  saved item.

### Changed

- Held-out rows are left out of training, LLM batches, LLM prompt examples,
  `GET /projects/<id>/tasks`, suggestion lists, confirm and bulk accept. The
  holdout is not drawn from rows an LLM touched, and only human or file labels
  grade a model.
- With a loop started, creating or training any model on the dataset keeps the
  minimum per label, up to the cap.
- The trainer no longer fails when a class has no row on the test side, and the
  confusion matrix keeps a row for it.
- The stopped-training message says what to do next, in plain words.

## [0.3.0] - 2026-10-30

### Fixed

- Annotation queue: tasks are created for rows with text and a blank label, so
  the queue is no longer always empty. Summarization is not queued, and saving
  an empty label is rejected. Saved annotations reach training for
  classification-family tasks.
- Training start: one start path. `/training/start` no longer returns a 500 on
  Postgres, duplicate starts return the active job, a failed start marks its job
  failed, and stale pending jobs are cleared on read.

### Added

- LLM annotation under `/api/llm`: suggestions for a batch, standalone annotation
  jobs on the worker (new `llm_queue`), agreement check and bulk accept. One
  OpenAI-compatible adapter. Suggestions become labels only through a human or a
  passed agreement check.
- Access: host-injected `FEATURE_GATE` (fails closed without `email_verified`),
  encrypted user keys (BYOK), per-project acknowledgement of the LLM destination,
  and `LLM_AUDIT_HOOK`.
- `USAGE_METER` seam with a default per-month DB counter, and
  `run_ml_worker(usage_meter=...)`.
- `purge_owner_llm_data` for hosts' account erasure.
- `label_source` on annotation tasks. Migration `lmc_0002_llm`.
- FAIR4RS metadata: `LICENSE` (MIT), `CITATION.cff`, `codemeta.json` and
  `.zenodo.json`, plus a "How to cite" section in the README.
- `CHANGELOG.md` and `CONTRIBUTING.md`.
- GitHub Actions CI workflow running the pytest suite on push and pull request.
- `docs/openapi.yaml`: OpenAPI 3.1 description of every HTTP route the
  blueprints register, with a test that fails if a registered route is missing
  from the spec.
- `docs/data-formats.md`: dataset upload format per task type and how trained
  models are stored and exported.
- `pyproject.toml`: authors, keywords, classifiers and project URLs; `PyYAML`
  in the `test` extra.

### Fixed

- `tests/test_datasets_routes.py` no longer imports the removed
  `flask_jwt_extended`, so the suite passes in a clean environment.

## [0.2.0] - 2026-10-29

### Added

- `AUTH_MODE="platform"`: the host signs one short-lived HS256 token per
  forwarded request (string `sub`, `iat`, `exp`, `exp - iat` at most 60 s);
  the token `sub` is the owner. The secret comes from
  `LEGAL_ML_PLATFORM_SECRET` or `LEGAL_ML_PLATFORM_SECRET_FILE` and is read per
  request; a missing secret returns 503 naming where it was expected. Bad
  tokens return 401. `NoOwnerRef` is refused in this mode (#31).
- Worker liveness: a per-job heartbeat thread (every 15 s) and a process-wide
  `legal_ml_worker_alive` key. `/training/status` and `/training/jobs` mark a
  running job with no beat for `LEGAL_ML_HEARTBEAT_STALE_SECONDS` (default
  120) as failed with a plain-language message, and queued jobs report
  "Waiting: the training service isn't running" when the worker is stale.
  Timings are configurable with `LivenessTimings` (#29).
- `job_state.transition`: every guarded job-status write is logged with job id
  and owner (#29).

### Changed

- Stop really stops training: `/training/stop/<job_id>` sets a Redis cancel
  flag and cancels through a guarded update. The worker checks the flag at
  pickup, between preprocessing, fit and save, and on every HuggingFace
  Trainer step; a cancelled run skips evaluation and save and removes any
  written model file. A cancelled job never becomes completed or failed (#28).
- `/training/status`: job status and error message always come from the
  database; Redis only refines progress and live metrics of active jobs.
- `/training/start` returns 409 "This model is already training" when the
  model already has a pending or running job, and locks the model row
  (`SELECT ... FOR UPDATE`) so concurrent starts serialise (#30).
- Logging replaces `print` in the worker.

### Fixed

- Annotation queue, counts, listing and reset are scoped to the dataset
  owner; a project id alone is no longer treated as an ownership check, and a
  dataset filter the caller does not own returns 404 (#27).
- Worker loops survive errors: a malformed or unloadable task fails its job
  with "Training could not start. Try again." and the loop keeps serving.
  Redis connection errors back off from 1 s to 30 s with one log line when
  Redis goes down and one when it returns (#30).
- A failed cancel-flag write during Stop is logged and the stop still
  succeeds; Redis errors during a job end it as failed instead of leaving it
  running.

## [0.1.0] - 2026-10-21

Initial extraction from the `reducing_digital_divide` (BAIGLE) monorepo. This
version was not tagged; it corresponds to commit `02c5b61`.

### Added

- Flask blueprints for datasets, models, training, visualization, EDA and
  annotation, registered by `init_ml_feature`, and `run_ml_worker` for the
  training, prediction and explanation queue consumers.
- Keycloak bearer-token validation and `AUTH_MODE` with a guarded `"none"`.
- Pluggable ownership through an opaque `owner_id` (`KeycloakOwnerRef`,
  `HostModelOwnerRef`, `NoOwnerRef`) and optional projects (`PROJECT_SCOPE`).
- Shared declarative base `MLBase` so the host owns the SQLAlchemy handle.
- Package-owned migrations, pluggable storage backend and configurable queue
  names.
- Probability calibration, paginated dataset rows, per-class keywords, the
  upload task-type catalogue, and experiment-monitoring metric points.

### Fixed

- LIME explanation `KeyError` for predicted classes other than index 1.
- Worker foreign-key resolution and the missing `psycopg2` dependency.
- Model/task compatibility checks, live progress and failure reasons.

[Unreleased]: https://github.com/shashankmc/legal-ml-core/compare/v0.22.1...HEAD
[0.22.1]: https://github.com/shashankmc/legal-ml-core/compare/v0.22.0...v0.22.1
[0.22.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.14.1...v0.15.0
[0.14.1]: https://github.com/shashankmc/legal-ml-core/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/shashankmc/legal-ml-core/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.11.2...v0.12.0
[0.11.2]: https://github.com/shashankmc/legal-ml-core/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/shashankmc/legal-ml-core/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/shashankmc/legal-ml-core/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/shashankmc/legal-ml-core/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shashankmc/legal-ml-core/compare/02c5b61...v0.2.0
[0.1.0]: https://github.com/shashankmc/legal-ml-core/commits/02c5b61
