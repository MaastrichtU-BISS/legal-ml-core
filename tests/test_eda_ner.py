"""Explore data for NER datasets: entity types, counts and the rows behind them
(shashankmc/reducing_digital_divide#179)."""
import json

import pandas as pd
import pytest

from legal_ml_core import Dataset, db
from tests.conftest import TEST_USER_ID


def _span(text, entity, kind, after=0):
    start = text.index(entity, after)
    return {"text": entity, "start": start, "end": start + len(entity), "type": kind}


def _wikiann_rows():
    """WikiANN-shaped rows: PER, ORG and LOC spans, a row with ``[]``, a blank
    cell, and a row naming the same person twice."""
    rows = []

    text = "Angela Merkel visited Paris ."
    rows.append((text, [_span(text, "Angela Merkel", "PER"), _span(text, "Paris", "LOC")]))
    text = "Siemens opened an office in Paris ."
    rows.append((text, [_span(text, "Siemens", "ORG"), _span(text, "Paris", "LOC")]))
    text = "Paris is in France ."
    rows.append((text, [_span(text, "Paris", "LOC"), _span(text, "France", "LOC")]))
    rows.append(("No entities here .", []))
    rows.append(("A blank cell .", None))
    text = "Angela Merkel met Angela Merkel ."
    rows.append(
        (text, [_span(text, "Angela Merkel", "PER"), _span(text, "Angela Merkel", "PER", 1)])
    )
    return rows


def _add_dataset(app, path, task_type="ner"):
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name=path.name,
            file_path=str(path),
            row_count=0,
            columns=[],
            task_type=task_type,
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def _csv_dataset(app, tmp_path, rows, name="wikiann.csv"):
    """Entities cells as JSON text, as the library import writes them."""
    path = tmp_path / name
    pd.DataFrame(
        {
            "text": [text for text, _ in rows],
            "entities": [
                spans if isinstance(spans, str) else ("" if spans is None else json.dumps(spans))
                for _, spans in rows
            ],
        }
    ).to_csv(path, index=False)
    return _add_dataset(app, path)


def _ner_stats(client, auth_headers, dataset_id):
    resp = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers)
    assert resp.status_code == 200
    return resp.get_json()["task_specific_analysis"]


def _wikiann_expectations(stats):
    assert stats["entity_column"] == "entities"
    assert stats["total_entities"] == 8
    assert stats["entity_types"] == [
        {
            "type": "LOC",
            "count": 4,
            "share": 0.5,
            "rows": 3,
            "mean_length_words": 1.0,
            "top_entities": [
                {"text": "Paris", "count": 3, "rows": 3},
                {"text": "France", "count": 1, "rows": 1},
            ],
        },
        {
            "type": "PER",
            "count": 3,
            "share": 0.375,
            "rows": 2,
            "mean_length_words": 2.0,
            "top_entities": [{"text": "Angela Merkel", "count": 3, "rows": 2}],
        },
        {
            "type": "ORG",
            "count": 1,
            "share": 0.125,
            "rows": 1,
            "mean_length_words": 1.0,
            "top_entities": [{"text": "Siemens", "count": 1, "rows": 1}],
        },
    ]
    assert stats["rows_without_entities"]["count"] == 2
    assert stats["rows_without_entities"]["percent"] == pytest.approx(100 * 2 / 6)
    assert stats["entities_per_row"] == {
        "mean": pytest.approx(8 / 6),
        "median": 2.0,
        "max": 2,
    }
    assert stats["rare_entity_types"] == []
    assert stats["unreadable_rows"] == 0
    assert stats["entity_message"] is None


def test_wikiann_span_json_lists_each_type_with_its_counts(client, auth_headers, app, tmp_path):
    dataset_id = _csv_dataset(app, tmp_path, _wikiann_rows())

    stats = _ner_stats(client, auth_headers, dataset_id)

    _wikiann_expectations(stats)
    assert stats["text_length_stats"]["max"] == len("Siemens opened an office in Paris .")
    assert "word_count_stats" in stats


def test_a_json_file_of_list_of_dict_spans_reads_the_same(client, auth_headers, app, tmp_path):
    path = tmp_path / "wikiann.json"
    records = [
        {"text": text, "entities": spans if spans is not None else ""}
        for text, spans in _wikiann_rows()
    ]
    path.write_text(json.dumps(records))
    dataset_id = _add_dataset(app, path)

    _wikiann_expectations(_ner_stats(client, auth_headers, dataset_id))


def test_only_empty_entity_lists_report_every_row_without_entities(
    client, auth_headers, app, tmp_path
):
    dataset_id = _csv_dataset(app, tmp_path, [("First row .", []), ("Second row .", [])])

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["entity_column"] == "entities"
    assert stats["total_entities"] == 0
    assert stats["entity_types"] == []
    assert stats["rows_without_entities"] == {"count": 2, "percent": 100.0}
    assert stats["entities_per_row"] == {"mean": 0.0, "median": 0.0, "max": 0}
    assert stats["unreadable_rows"] == 0
    assert stats["entity_message"] == "No row has any entities."


def test_malformed_cells_are_counted_as_unreadable_not_a_500(client, auth_headers, app, tmp_path):
    text = "Paris is in France ."
    rows = [
        (text, [_span(text, "Paris", "LOC")]),
        (text, '[{"text": "Paris", "start": 0'),  # cut-off JSON
        (text, [{"text": "Paris", "start": 0, "end": 99, "type": "LOC"}]),  # past the text
        (text, [{"text": "Paris", "start": 0, "end": 5}]),  # no type
        (text, "LOC, ORG"),  # a type list is not spans, and not unreadable
    ]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["unreadable_rows"] == 3
    assert stats["total_entities"] == 1
    assert [entry["type"] for entry in stats["entity_types"]] == ["LOC"]
    # The readable row with an entity and the type-list row; unreadable rows
    # are reported on their own.
    assert stats["rows_without_entities"]["count"] == 1
    assert stats["entity_message"] is None


def test_a_column_of_only_malformed_cells_says_so(client, auth_headers, app, tmp_path):
    rows = [("Paris .", "[not json"), ("Rome .", '[{"start": 0, "end": 4}]')]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["unreadable_rows"] == 2
    assert stats["entity_types"] == []
    assert stats["entity_message"].startswith("No row's entities could be read.")


def test_no_entities_column_answers_a_message_with_every_key(client, auth_headers, app, tmp_path):
    path = tmp_path / "plain.csv"
    pd.DataFrame({"text": ["Paris is in France .", "Rome ."]}).to_csv(path, index=False)
    dataset_id = _add_dataset(app, path)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["entity_column"] is None
    assert stats["total_entities"] == 0
    assert stats["entity_types"] == []
    assert stats["rows_without_entities"] == {"count": 0, "percent": 0.0}
    assert stats["entities_per_row"] == {"mean": 0.0, "median": 0.0, "max": 0}
    assert stats["rare_entity_types"] == []
    assert stats["unreadable_rows"] == 0
    assert "no entities column" in stats["entity_message"]
    assert "text_length_stats" in stats


def test_types_under_five_percent_are_rare_and_top_entities_stop_at_ten(
    client, auth_headers, app, tmp_path
):
    rows = []
    for i in range(24):
        text = f"City{i} ."
        rows.append((text, [_span(text, f"City{i}", "LOC")]))
    text = "Euro ."
    rows.append((text, [_span(text, "Euro", "MISC")]))
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["rare_entity_types"] == ["MISC"]
    loc = stats["entity_types"][0]
    assert loc["type"] == "LOC"
    assert len(loc["top_entities"]) == 10


def test_visualisations_have_an_entities_per_type_bar_chart(client, auth_headers, app, tmp_path):
    dataset_id = _csv_dataset(app, tmp_path, _wikiann_rows())

    resp = client.get(f"/api/datasets/{dataset_id}/eda/visualizations", headers=auth_headers)

    assert resp.status_code == 200
    chart = json.loads(resp.get_json()["entities_per_type"])
    bar = chart["data"][0]
    assert bar["type"] == "bar"
    assert list(bar["x"]) == ["LOC", "PER", "ORG"]
    assert list(bar["y"]) == [4, 3, 1]


def test_a_dataset_without_entities_has_no_entities_chart(client, auth_headers, app, tmp_path):
    path = tmp_path / "plain.csv"
    pd.DataFrame({"text": ["Paris ."]}).to_csv(path, index=False)
    dataset_id = _add_dataset(app, path)

    resp = client.get(f"/api/datasets/{dataset_id}/eda/visualizations", headers=auth_headers)

    assert resp.status_code == 200
    assert "entities_per_type" not in resp.get_json()


@pytest.mark.parametrize("name", ["wikiann.csv", "wikiann.json"])
def test_rows_filtered_by_entity_total_the_rows_eda_shows(
    client, auth_headers, app, tmp_path, name
):
    """Opening a top entity, or a type, in Browse data shows as many rows as the
    analysis says."""
    if name.endswith(".csv"):
        dataset_id = _csv_dataset(app, tmp_path, _wikiann_rows(), name=name)
    else:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                [{"text": t, "entities": s if s is not None else ""} for t, s in _wikiann_rows()]
            )
        )
        dataset_id = _add_dataset(app, path)
    stats = _ner_stats(client, auth_headers, dataset_id)

    for entry in stats["entity_types"]:
        by_type = client.get(
            f"/api/datasets/{dataset_id}/rows",
            query_string={"entity_type": entry["type"]},
            headers=auth_headers,
        ).get_json()
        assert by_type["total_rows"] == entry["rows"]
        for top in entry["top_entities"]:
            body = client.get(
                f"/api/datasets/{dataset_id}/rows",
                query_string={"entity": top["text"], "entity_type": entry["type"]},
                headers=auth_headers,
            ).get_json()
            assert body["total_rows"] == top["rows"]

    paris = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"entity": "Paris", "entity_type": "LOC"},
        headers=auth_headers,
    ).get_json()
    assert [row["text"] for row in paris["rows"]] == [
        "Angela Merkel visited Paris .",
        "Siemens opened an office in Paris .",
        "Paris is in France .",
    ]
    # Same text, wrong type: nothing.
    none = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"entity": "Paris", "entity_type": "PER"},
        headers=auth_headers,
    ).get_json()
    assert none["total_rows"] == 0


def test_rows_entity_filter_pages_and_combines_with_term(client, auth_headers, app, tmp_path):
    dataset_id = _csv_dataset(app, tmp_path, _wikiann_rows())

    body = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"entity": "Paris", "entity_type": "LOC", "term": "siemens", "limit": 1},
        headers=auth_headers,
    ).get_json()

    assert body["total_rows"] == 1
    assert body["rows"][0]["text"] == "Siemens opened an office in Paris ."
    assert body["has_more"] is False


def test_rows_entity_filter_without_an_entities_column_is_a_400(
    client, auth_headers, app, tmp_path
):
    path = tmp_path / "plain.csv"
    pd.DataFrame({"text": ["Paris ."]}).to_csv(path, index=False)
    dataset_id = _add_dataset(app, path)

    resp = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"entity": "Paris", "entity_type": "LOC"},
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert "no entities column" in resp.get_json()["error"]
