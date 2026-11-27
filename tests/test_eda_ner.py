"""Explore data for NER datasets: entity types, counts and the rows behind them
(shashankmc/reducing_digital_divide#179)."""
import json

import numpy as np
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


# --- council round 1 --------------------------------------------------------


def _generic_rows():
    """WikiANN rows with a numeric ``page`` column and one blank ``source``."""
    rows = _wikiann_rows()
    return [
        {
            "text": text,
            "entities": spans if spans is not None else "",
            "page": i + 1,
            "source": "" if i == 2 else "wiki",
        }
        for i, (text, spans) in enumerate(rows)
    ]


def _generic_dataset(app, tmp_path, suffix):
    records = _generic_rows()
    path = tmp_path / f"generic{suffix}"
    if suffix == ".csv":
        frame = pd.DataFrame(records)
        frame["entities"] = [
            json.dumps(cell) if isinstance(cell, list) else cell for cell in frame["entities"]
        ]
        frame.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(records))
    return _add_dataset(app, path)


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_generic_eda_of_an_ner_file_keeps_nulls_numbers_and_quality(
    client, auth_headers, app, tmp_path, suffix
):
    """The generic sections read the file with pandas' usual types, and only
    the span work reads it as written (perform_eda_analysis' exact_df)."""
    dataset_id = _generic_dataset(app, tmp_path, suffix)

    resp = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers)
    assert resp.status_code == 200
    analysis = resp.get_json()

    assert analysis["data_quality"]["overall_quality_score"] > 0
    assert analysis["column_analysis"]
    if suffix == ".csv":
        # A blank cell is a null in the generic sections (pd.read_json reads
        # "" as text, as it always has).
        assert analysis["column_analysis"]["source"]["null_count"] == 1
    assert "page" in analysis["statistical_summary"]["numeric_summary"]
    _wikiann_expectations(analysis["task_specific_analysis"])

    resp = client.get(f"/api/datasets/{dataset_id}/eda/visualizations", headers=auth_headers)
    assert resp.status_code == 200
    charts = resp.get_json()
    assert "numeric_distributions" in charts
    assert "missing_data" in charts
    assert list(json.loads(charts["entities_per_type"])["data"][0]["x"]) == ["LOC", "PER", "ORG"]


def test_exact_df_defaults_to_df_for_in_memory_callers():
    from legal_ml_core.routes.eda import generate_visualizations, perform_eda_analysis

    df = pd.DataFrame(
        {"text": [t for t, _ in _wikiann_rows()], "entities": [s for _, s in _wikiann_rows()]}
    )
    df["entities"] = [json.dumps(s) if s is not None else "" for s in df["entities"]]

    _wikiann_expectations(perform_eda_analysis(df, "ner")["task_specific_analysis"])
    assert "entities_per_type" in generate_visualizations(df, "ner")


def test_the_entities_column_is_not_a_categorical_column(client, auth_headers, app, tmp_path):
    path = tmp_path / "sparse.csv"
    text = "Paris ."
    cells = ["[]"] * 8 + ["", json.dumps([_span(text, "Paris", "LOC")])]
    pd.DataFrame({"text": [f"Row {i} ." for i in range(9)] + [text], "entities": cells}).to_csv(
        path, index=False
    )
    dataset_id = _add_dataset(app, path)

    resp = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers)

    assert resp.status_code == 200
    assert "entities" not in resp.get_json()["statistical_summary"]["categorical_summary"]


def test_rows_without_entities_percent_leaves_out_unreadable_rows(
    client, auth_headers, app, tmp_path
):
    text = "Paris ."
    rows = [
        (text, [_span(text, "Paris", "LOC")]),
        (text, []),
        (text, "[not json"),
        (text, "[not json either"),
    ]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["rows_without_entities"] == {"count": 1, "percent": 50.0}


def test_a_column_of_type_lists_says_the_entities_cannot_be_counted(
    client, auth_headers, app, tmp_path
):
    rows = [("Merkel visited Paris .", "PER, LOC"), ("Siemens .", "ORG"), ("Blank .", None)]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["total_entities"] == 0
    assert stats["unreadable_rows"] == 0
    assert "names entity types but not where" in stats["entity_message"]
    assert "cannot be counted" in stats["entity_message"]


def test_empty_rows_and_one_malformed_row_do_not_say_nothing_could_be_read(
    client, auth_headers, app, tmp_path
):
    rows = [(f"Row {i} .", []) for i in range(9)] + [("Bad .", "[not json")]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["unreadable_rows"] == 1
    assert stats["entity_message"] == (
        "No readable row has any entities; 1 row could not be read."
    )


def test_rows_of_an_ner_dataset_show_and_search_cells_verbatim(
    client, auth_headers, app, tmp_path
):
    text = "NA filed case 00123 ."
    path = tmp_path / "verbatim.csv"
    pd.DataFrame(
        {
            "text": [text, "Other ."],
            "entities": [json.dumps([_span(text, "00123", "MISC")]), "[]"],
            "code": ["NA", "00123"],
        }
    ).to_csv(path, index=False)
    dataset_id = _add_dataset(app, path)

    body = client.get(f"/api/datasets/{dataset_id}/rows", headers=auth_headers).get_json()
    assert [row["code"] for row in body["rows"]] == ["NA", "00123"]

    found = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"term": "00123", "entity": "00123"},
        headers=auth_headers,
    ).get_json()
    assert found["total_rows"] == 1
    assert found["rows"][0]["text"] == text


def test_rows_entity_filter_keeps_the_right_rows_of_a_labelled_json_index(
    client, auth_headers, app, tmp_path
):
    first, second = "Paris is big .", "Rome is old ."
    path = tmp_path / "labelled.json"
    path.write_text(
        json.dumps(
            {
                "text": {"a": first, "b": second},
                "entities": {
                    "a": [_span(first, "Paris", "LOC")],
                    "b": [_span(second, "Rome", "LOC")],
                },
            }
        )
    )
    dataset_id = _add_dataset(app, path)

    body = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"entity": "Rome"},
        headers=auth_headers,
    ).get_json()

    assert body["total_rows"] == 1
    assert body["rows"][0]["text"] == second


def test_rows_entity_filter_without_a_text_column_is_a_clear_400(
    client, auth_headers, app, tmp_path
):
    path = tmp_path / "no_text.csv"
    pd.DataFrame({"body": ["Paris ."], "entities": ["[]"]}).to_csv(path, index=False)
    dataset_id = _add_dataset(app, path)

    resp = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"entity_type": "LOC"},
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert "no text column" in resp.get_json()["error"]


def test_numpy_integer_offsets_are_read_as_spans():
    import numpy as np

    from legal_ml_core.annotation_queue import read_span_cell

    text = "Paris is big ."
    cell = [{"text": "Paris", "start": np.int64(0), "end": np.int64(5), "type": "LOC"}]

    spans, unreadable = read_span_cell(cell, text)

    assert unreadable is False
    assert spans == [{"text": "Paris", "start": 0, "end": 5, "type": "LOC"}]
    assert type(spans[0]["start"]) is int


def test_blank_rows_and_one_malformed_row_do_not_say_nothing_could_be_read(
    client, auth_headers, app, tmp_path
):
    rows = [(f"Row {i} .", None) for i in range(9)] + [("Bad .", "[not json")]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert stats["unreadable_rows"] == 1
    assert stats["entity_message"] == (
        "No readable row has any entities; 1 row could not be read."
    )


def test_type_lists_beside_an_empty_row_still_say_the_entities_cannot_be_counted(
    client, auth_headers, app, tmp_path
):
    rows = [("Empty .", [])] + [(f"Row {i} .", "PER, LOC") for i in range(9)]
    dataset_id = _csv_dataset(app, tmp_path, rows)

    stats = _ner_stats(client, auth_headers, dataset_id)

    assert "names entity types but not where" in stats["entity_message"]


def test_pandas_missing_values_are_blank_cells_not_type_lists():
    from legal_ml_core.routes.eda import _is_blank_cell

    for value in (None, float("nan"), np.float32("nan"), pd.NA, pd.NaT, "  "):
        assert _is_blank_cell(value)
    assert not _is_blank_cell("PER, LOC")
    assert not _is_blank_cell([])


def test_a_json_column_mixing_lists_and_numbers_is_not_a_500(
    client, auth_headers, app, tmp_path
):
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            [
                {"text": "Paris .", "entities": [_span("Paris .", "Paris", "LOC")], "meta": {"src": "wiki"}},
                {"text": "Rome .", "entities": 0, "meta": 42},
                {"text": "Oslo .", "entities": [], "meta": "news"},
            ]
        )
    )
    dataset_id = _add_dataset(app, path)

    resp = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers)

    assert resp.status_code == 200
    stats = resp.get_json()["task_specific_analysis"]
    assert stats["total_entities"] == 1


def test_mixed_key_types_become_json_text_keys():
    from legal_ml_core.routes.eda import convert_numpy_types

    out = convert_numpy_types({1: "a", "b": 2, None: 4, np.int64(5): 6})
    assert out == {"1": "a", "b": 2, "null": 4, "5": 6}
    assert convert_numpy_types({True: 1, "x": 2}) == {"true": 1, "x": 2}


def test_an_entity_filter_that_empties_the_rows_keeps_the_columns(
    client, auth_headers, app, tmp_path
):
    dataset_id = _csv_dataset(app, tmp_path, _wikiann_rows())

    body = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"term": "zzzunseen", "entity": "Paris"},
        headers=auth_headers,
    ).get_json()

    assert body["total_rows"] == 0
    assert body["columns"] == ["text", "entities"]


def test_an_entity_filter_on_a_file_with_no_rows_keeps_the_columns(
    client, auth_headers, app, tmp_path
):
    path = tmp_path / "header_only.csv"
    path.write_text("text,entities\n")
    dataset_id = _add_dataset(app, path)

    resp = client.get(
        f"/api/datasets/{dataset_id}/rows",
        query_string={"term": "paris", "entity": "Paris"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert resp.get_json()["columns"] == ["text", "entities"]
