"""Every reader of training labels reads them as the trainer does (BAIGLE #218)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from legal_ml_core.annotation_format import label_text
from legal_ml_core.annotation_queue import holdout_examples


@pytest.mark.parametrize(
    "value, text",
    [
        (1, "1"),
        (np.int64(2), "2"),
        (1.0, "1"),
        (np.float64(1.0), "1"),
        (np.float32(1.0), "1"),
        (1.5, "1.5"),
        ("1", "1"),
        ("007", "007"),
        ("1.50", "1.50"),
        (" a ", " a "),
        (True, "True"),
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
        ("  ", None),
    ],
)
def test_label_text(value, text):
    assert label_text(value) == text


def test_held_out_file_labels_grade_against_the_trainers_classes():
    """A held-out row whose file label reads as 1.0 grades against the class "1"."""
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0.0, 1.0, None]})
    dataset = SimpleNamespace(
        label_column="label",
        task_type="classification",
        text_column="text",
        annotation_tasks=[],
        al_config=SimpleNamespace(holdout_indices=[0, 1, 2]),
    )

    texts, labels = holdout_examples(df, dataset)

    assert texts == ["a", "b"]
    assert labels == ["0", "1"]
