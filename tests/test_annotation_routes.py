import json

from legal_ml_core import AnnotationTask, Dataset, db
from tests.conftest import TEST_USER_ID


def _make_project_and_dataset(app, project_id="proj-1", owner_id=TEST_USER_ID):
    """Projects are a host concern; register this (project, owner) pair with the
    conftest project validator, then create the dataset/task it scopes.
    """
    with app.app_context():
        app.config.setdefault("_known_projects", set()).add((project_id, owner_id))
        dataset = Dataset(
            owner_id=owner_id,
            name="Annotated dataset",
            file_path="/tmp/a.csv",
            row_count=2,
            columns=[],
            task_type="classification",
            project_id=project_id,
        )
        db.session.add(dataset)
        db.session.flush()
        task = AnnotationTask(dataset_id=dataset.id, item_index=0, text="clause text", status="pending")
        db.session.add(task)
        db.session.commit()
        return project_id, dataset.id, task.id


def test_get_next_annotation_task_and_save(client, auth_headers, app):
    project_id, dataset_id, task_id = _make_project_and_dataset(app)

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/next", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["task"]["id"] == task_id
    assert body["stats"]["total"] == 1

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save",
        json={"annotation_label": "positive"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.get_json()["task"]["status"] == "annotated"

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/next", headers=auth_headers)
    assert resp.get_json()["task"] is None
    assert "All items annotated" in resp.get_json()["message"]


def test_skip_annotation(client, auth_headers, app):
    _, _, task_id = _make_project_and_dataset(app)

    resp = client.post(f"/api/annotation/tasks/{task_id}/skip", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["task"]["status"] == "skipped"


def test_annotation_queue_requires_owned_project(client, auth_headers, app):
    project_id, _, _ = _make_project_and_dataset(app, project_id="proj-2", owner_id="someone-else")

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/next", headers=auth_headers)
    assert resp.status_code == 404


def test_reset_annotation_tasks(client, auth_headers, app):
    project_id, _, task_id = _make_project_and_dataset(app, project_id="proj-3")

    client.post(f"/api/annotation/tasks/{task_id}/save", json={"annotation_label": "x"}, headers=auth_headers)

    resp = client.post(f"/api/annotation/projects/{project_id}/tasks/reset", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["reset_count"] == 1

    resp = client.get(f"/api/annotation/tasks/{task_id}", headers=auth_headers)
    assert resp.get_json()["task"]["status"] == "pending"


OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"


def _headers_for(sub):
    return {"Authorization": f"Bearer test-token::{json.dumps({'sub': sub})}"}


def _foreign_dataset_on_project(app, project_id):
    """Student B's dataset that references student A's project id."""
    with app.app_context():
        dataset = Dataset(
            owner_id=OTHER_USER_ID,
            name="B's dataset",
            file_path="/tmp/b.csv",
            row_count=3,
            columns=[],
            task_type="classification",
            project_id=project_id,
        )
        db.session.add(dataset)
        db.session.flush()
        for i, status in enumerate(["pending", "annotated", "annotated"]):
            db.session.add(
                AnnotationTask(dataset_id=dataset.id, item_index=i, text=f"B's document {i}", status=status)
            )
        db.session.commit()
        return dataset.id


def test_queue_next_rejects_other_users_dataset_filter(client, auth_headers, app):
    project_id, _, _ = _make_project_and_dataset(app, project_id="proj-a")
    foreign_dataset_id = _foreign_dataset_on_project(app, project_id)

    resp = client.get(
        f"/api/annotation/projects/{project_id}/queue/next?dataset={foreign_dataset_id}",
        headers=auth_headers,
    )
    assert resp.status_code == 404
    body = resp.get_json()
    assert "stats" not in body
    assert "task" not in body


def test_queue_next_serves_and_counts_only_own_tasks(client, auth_headers, app):
    project_id, own_dataset_id, own_task_id = _make_project_and_dataset(app, project_id="proj-a")
    _foreign_dataset_on_project(app, project_id)

    for url in (
        f"/api/annotation/projects/{project_id}/queue/next",
        f"/api/annotation/projects/{project_id}/queue/next?dataset={own_dataset_id}",
    ):
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["task"]["id"] == own_task_id
        assert body["stats"] == {"total": 1, "annotated": 0, "percentage": 0}


def test_queue_stats_count_only_own_tasks(client, auth_headers, app):
    project_id, _, _ = _make_project_and_dataset(app, project_id="proj-a")
    _foreign_dataset_on_project(app, project_id)

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/stats", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["stats"] == {
        "total": 1,
        "pending": 1,
        "annotated": 0,
        "skipped": 0,
        "percentage_complete": 0,
    }


def test_list_tasks_returns_only_own_tasks(client, auth_headers, app):
    project_id, _, own_task_id = _make_project_and_dataset(app, project_id="proj-a")
    _foreign_dataset_on_project(app, project_id)

    resp = client.get(f"/api/annotation/projects/{project_id}/tasks", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert [t["id"] for t in body["tasks"]] == [own_task_id]
    assert body["count"] == 1


def test_reset_leaves_other_users_tasks_untouched(client, auth_headers, app):
    project_id, _, _ = _make_project_and_dataset(app, project_id="proj-a")
    foreign_dataset_id = _foreign_dataset_on_project(app, project_id)

    resp = client.post(f"/api/annotation/projects/{project_id}/tasks/reset", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["reset_count"] == 1

    # Student B, who also files work under this project id, still sees their annotations.
    app.config["_known_projects"].add((project_id, OTHER_USER_ID))
    resp = client.get(
        f"/api/annotation/projects/{project_id}/queue/stats", headers=_headers_for(OTHER_USER_ID)
    )
    assert resp.get_json()["stats"]["annotated"] == 2


def _add_tasks(app, dataset_id, rows):
    """Add tasks to ``dataset_id``: one per (status, label, is_holdout) row."""
    with app.app_context():
        start = db.session.query(db.func.count()).filter(AnnotationTask.dataset_id == dataset_id).scalar()
        for offset, (status, label, holdout) in enumerate(rows):
            db.session.add(
                AnnotationTask(
                    dataset_id=dataset_id,
                    item_index=start + offset,
                    text=f"item {start + offset}",
                    status=status,
                    annotation_label=label,
                    is_holdout=holdout,
                )
            )
        db.session.commit()


def _stats(client, project_id, headers, dataset_id=None):
    url = f"/api/annotation/projects/{project_id}/queue/stats"
    if dataset_id is not None:
        url += f"?dataset_id={dataset_id}"
    resp = client.get(url, headers=headers)
    assert resp.status_code == 200
    return resp.get_json()["stats"]


def test_queue_stats_count_confirmed_labels_per_label_in_one_query(client, auth_headers, app):
    from sqlalchemy import event

    project_id, dataset_id, _ = _make_project_and_dataset(app, project_id="proj-a")
    _add_tasks(app, dataset_id, [
        ("annotated", "positive", False),
        ("annotated", "positive", False),
        ("annotated", "negative", False),
        # Held out, pending, skipped and blank-labelled rows count nothing.
        ("annotated", "positive", True),
        ("annotated", "neutral", True),
        ("pending", "negative", False),
        ("skipped", "negative", False),
        ("annotated", "", False),
        ("annotated", "   ", False),
        ("annotated", None, False),
    ])
    # Another of the caller's Datasets on the same project does not count.
    _, other_dataset_id, _ = _make_project_and_dataset(app, project_id="proj-a")
    _add_tasks(app, other_dataset_id, [("annotated", "positive", False), ("annotated", "other", False)])

    with app.app_context():
        engine = db.engine
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        stats = _stats(client, project_id, auth_headers, dataset_id)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert stats["label_counts"] == {"positive": 2, "negative": 1}
    grouped = [s for s in statements if "GROUP BY" in s.upper()]
    assert len(grouped) == 1
    # The project-wide counts keep their meaning: they are not scoped to the Dataset.
    assert stats["total"] == 1 + 10 + 1 + 2
    assert stats["annotated"] == 8 + 2


def test_queue_stats_without_dataset_id_have_no_label_counts(client, auth_headers, app):
    project_id, dataset_id, _ = _make_project_and_dataset(app, project_id="proj-a")
    _add_tasks(app, dataset_id, [("annotated", "positive", False)])

    assert _stats(client, project_id, auth_headers) == {
        "total": 2,
        "pending": 1,
        "annotated": 1,
        "skipped": 0,
        "percentage_complete": 50,
    }


def test_queue_stats_label_counts_empty_for_unowned_or_off_project_dataset(client, auth_headers, app):
    project_id, _, _ = _make_project_and_dataset(app, project_id="proj-a")
    foreign_dataset_id = _foreign_dataset_on_project(app, project_id)
    _add_tasks(app, foreign_dataset_id, [("annotated", "positive", False)])
    _, off_project_dataset_id, _ = _make_project_and_dataset(app, project_id="proj-b")
    _add_tasks(app, off_project_dataset_id, [("annotated", "positive", False)])

    for dataset_id in (foreign_dataset_id, off_project_dataset_id, "no-such-dataset"):
        stats = _stats(client, project_id, auth_headers, dataset_id)
        assert stats["label_counts"] == {}
        assert stats["total"] == 1
