# legal-ml-core

Core ML feature for BAIGLE-style legal-ML apps — dataset management, model
training/prediction/explanation, visualization, EDA, annotation, and
experiment monitoring — as an installable Flask package, extracted from the
`reducing_digital_divide` monorepo.

This package is the single source of truth for the ML feature's SQLAlchemy
models and compute engine, shared between the Flask API process and a
separate worker process (previously two independent, drifting
representations of the same database schema).

## What's in scope

- Dataset upload/validation/preview/EDA
- Model catalog, configuration, training, prediction, LIME/SHAP explanation
- Visualization (performance metrics, confusion matrix, clustering, topic modeling, model comparison)
- Annotation task queue
- Summarization: extractive (sumy) and abstractive (HuggingFace) text
  summaries as a queued operation, folded in from `legal-summarizer-service`
- Experiment monitoring: per-training-job metric time series (dense for the
  HuggingFace Trainer path, coarse phase markers for classical sklearn
  algorithms) and cross-run hyperparameter-vs-performance comparison
- Pluggable ownership: opaque `owner_id` scoped by a Keycloak subject, a host
  resolver, or nothing (single-tenant) — no local user table required

Out of scope (stays in the host app): auth, generic project CRUD, and the
unrelated "learning platform" tutorials feature (including its sandboxed
code-execution/package-installation queues).

## Installing into a host Flask app

A host that has its own models owns the `SQLAlchemy` handle, built on this
package's shared `MLBase`:

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from legal_ml_core import MLBase, init_ml_feature
import redis

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://..."

db = SQLAlchemy(model_class=MLBase)   # host owns the handle, on the shared base
db.init_app(app)

# Host models subclass MLBase too, so they share one registry with the package:
#   from legal_ml_core import MLBase
#   class Project(MLBase): ...

redis_client = redis.Redis.from_url("redis://...")
init_ml_feature(app, redis_client, model_storage_path="/app/models", db=db)
```

An **ML-only host** with no models of its own can skip the handle and let the
package use its default one (pass `db=None`, or just call
`from legal_ml_core import db; db.init_app(app)` — `db` is a proxy onto the
package's default handle):

```python
from legal_ml_core import db, init_ml_feature
db.init_app(app)
init_ml_feature(app, redis_client, model_storage_path="/app/models")  # db omitted
```

**Ownership is pluggable and needs no local user table.** ML rows carry an
opaque `owner_id`, resolved by an `OwnerRef` strategy. The default,
`KeycloakOwnerRef`, uses the token `sub` — so a host that authenticates with
Keycloak needs no `users`/`projects` tables at all:

```python
app.config["OWNER_REF"] = KeycloakOwnerRef()          # default; sub is the owner
# or: HostModelOwnerRef(lambda payload: my_resolver(payload))  # host user model
# or: NoOwnerRef()                                            # single-tenant, no auth
```

**Authentication is selected by `AUTH_MODE`:**

- `"keycloak"` (default) — verifies an RS/ES-signed Keycloak access token
  against `AUTH_ISSUER` / `AUTH_AUDIENCE` (JWKS from `AUTH_JWKS_URI` or the
  issuer). Never accepts HS256.
- `"platform"` — the host (e.g. Legal Blocks) signs one short-lived HS256 token
  per forwarded request; its `sub` is the owner, so each platform user's data
  is isolated without Keycloak. Tokens must carry a string `sub`, `iat` and
  `exp`, with `exp - iat` at most 60 seconds; RS256 and other algorithms are
  rejected. Bad, expired or missing tokens get a 401 and a log line.
  `OWNER_REF = NoOwnerRef()` is refused in this mode (no shared owner).
- `"none"` — no token; requires `LEGAL_ML_ALLOW_NO_AUTH = True`.

The platform secret's location is the host's concern; the package reads it at
request time (so the host may create it after this service starts):

```python
app.config["AUTH_MODE"] = "platform"
app.config["LEGAL_ML_PLATFORM_SECRET"] = "..."                 # the value, or
app.config["LEGAL_ML_PLATFORM_SECRET_FILE"] = "/run/ml-secret/secret"  # a path
```

The value wins over the file. If neither yields a secret the request gets a 503
naming where the secret was expected (e.g. `platform secret not found at
/run/ml-secret/secret`), so a deployment problem never looks like a sign-in
problem.

Projects are optional, selected by `PROJECT_SCOPE`:

- `"none"` (default) — a non-empty `project_id` is an opaque reference the host
  owns; the package never validates it.
- `"host_table"` — the host installs a `PROJECT_VALIDATOR(session, project_id,
  owner_id) -> bool` callable (BAIGLE's path; it has its own `projects` table).
- `"package_table"` — the package validates against its own optional
  `ml_projects` table (`MLProject`), for a self-contained host that wants
  grouping without bringing a model.

> The earlier hard foreign keys into `users`/`projects` (and the stub tables the
> worker used to fabricate) were removed in the reusability refactor; see
> `docs/legal-ml-core-reusability-plan.md` in the BAIGLE repo.

**Registry sharing contract.** All models — the package's and the host's — must
be attached to **one** declarative base: `legal_ml_core.db.MLBase`. Build your
handle as `SQLAlchemy(model_class=MLBase)` and subclass `MLBase` for your own
models; do **not** construct a second independent `SQLAlchemy()` on a different
base — two such handles never share a registry. `init_ml_feature` enforces this
by checking that the handle's metadata is `MLBase`'s. (Flask-SQLAlchemy derives
its own base class from `model_class`, so `db.Model is not MLBase`; the thing
that must match is the metadata, which is what the check compares.)

## Packaged into host images

Hosts commonly vendor this package into more than one image (e.g. a Flask API
image and a separate worker image). Because the package is copied at build time,
a change here only takes effect after **every** image that vendors it is
rebuilt — rebuilding just the API image leaves a stale worker running the old
models. Rebuild all of them (in BAIGLE, `docker compose build` with no service
argument).

## Running the worker

```python
from legal_ml_core import run_ml_worker

run_ml_worker(
    redis_url="redis://...",
    database_url="postgresql://...",
    model_storage_path="/app/models",
)
```

This starts the training/prediction/explanation/summarization queue-consumer
threads, and downloads NLTK's `punkt` data on start if it is missing. It
does **not** include the learning-platform's code-execution/package-install
queues — a host app composing its own worker process should run those
alongside this, in the same process, using its own code.

**Liveness.** While a job trains, the worker beats a per-job heartbeat key
every 15s, and a process-wide liveness key while it is up. The API treats a
`running` job with no beat for `LEGAL_ML_HEARTBEAT_STALE_SECONDS` (Flask config,
default 120) as dead and marks it failed, and tells queued jobs that the
training service isn't running when the worker key is that stale. The worker's
own window is `run_ml_worker(liveness=legal_ml_core.liveness.LivenessTimings(stale_after=...))`, also
120s by default; keep the two equal.

```python
app.config["LEGAL_ML_HEARTBEAT_STALE_SECONDS"] = 120  # default
```

## Summarization

Folded in from `legal-summarizer-service` (and its `vue-legal-summarizer`
frontend, now part of `vue-legal-ml-builder`). Same algorithms, run by this
package's worker on `summarization_queue`; job state is a Redis key
(`summary_status_{request_id}`, 1 h TTL), not a database row.

| Route | |
|---|---|
| `POST /api/summarize` | Queue a summary. Body: `input_text` (required), `summary_length` (`short`/`medium`/`long` or a target word count), `priority` (`relevance`, `readability`, each 0-1), `model`, optional `request_id`. Answers 202 with the job's status. |
| `GET /api/summarize/<request_id>` | Status: `queued` → `processing` → `complete` (with `summarized_text`) or `error`. Only the job's owner sees it; anyone else gets 404. A queued job carries `status_message` when no worker is running. |
| `GET /api/summarize/models` | The catalogue: extractive `lex_rank` (default), `lsa`, `text_rank`, `luhn`, `edmundson`; abstractive `hf:` models; whether abstractive is available. |

Clients poll; there is no SSE stream. Unknown models are rejected with 400
(the old service silently fell back to LexRank), and a request can only name
an `hf:` model the deployment offers, so no caller can make the worker
download an arbitrary hub model.

```python
app.config["LEGAL_ML_SUMMARIZATION_HF_MODELS"] = ["sshleifer/distilbart-cnn-12-6"]  # default: the three in compute/summarization.py; [] turns abstractive off
app.config["LEGAL_ML_SUMMARIZATION_MAX_INPUT_CHARS"] = 500_000                    # default
```

Abstractive models are downloaded into the Hugging Face cache on first use and
kept in memory by the worker. Training a summarization model is a separate
matter and is still refused (`UNSUPPORTED_TASK_TYPES`).

## Experiment monitoring limitation

Per-step training curves (`TrainingMetricPoint`, polled via
`GET /training/<job_id>/metrics`) are only dense/real for the HuggingFace
`Trainer` path (LegalBERT/RoBERTa fine-tuning), which has a genuine
epoch/step loop. Classical sklearn algorithms (SVM, decision tree, random
forest, naive Bayes, logistic regression, KMeans/agglomerative/DBSCAN,
LDA/NMF) fit synchronously with no epoch loop, so they only get coarse phase
markers (`preprocessing_started`, `preprocessing_done`, `training_started`,
`training_done`, `model_saved`). Cross-run hyperparameter-vs-performance
comparison (`GET /visualization/parameter-comparison`) works uniformly
across all model types since it's retrospective, reading already-stored
`Model.parameters`/`TrainingJob.metrics`.

## Development

```bash
pip install -e ".[test]"
pytest
```
