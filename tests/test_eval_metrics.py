from eval.metrics import confusion, precision_recall


def test_precision_recall():
    y_true = ["A", "A", "B", "B", "B"]
    y_pred = ["A", "B", "B", "B", "REJECTED"]
    m = precision_recall(y_true, y_pred, ["A", "B", "C"])
    assert m["per_label"]["A"]["precision"] == 1.0 and m["per_label"]["A"]["recall"] == 0.5
    assert abs(m["per_label"]["B"]["precision"] - 2 / 3) < 1e-9 and abs(m["per_label"]["B"]["recall"] - 2 / 3) < 1e-9
    assert m["per_label"]["C"]["precision"] is None and m["accuracy"] == 0.6
    assert confusion(y_true, y_pred)["B"] == {"B": 2, "REJECTED": 1}


def test_multilabel_counts_primary_and_secondary_categories():
    from eval.metrics import multilabel_pr
    true = [{"A", "B"}, {"A"}, {"C"}]
    pred = [{"A"}, {"A", "B"}, {"C"}]
    m = multilabel_pr(true, pred, ["A", "B", "C"])
    assert m["per_label"]["A"] == {"precision": 1.0, "recall": 1.0, "support": 2, "tp": 2, "fp": 0, "fn": 0}
    assert (m["per_label"]["B"]["precision"], m["per_label"]["B"]["recall"]) == (0.0, 0.0)  # missed once, invented once
    assert (m["micro_precision"], m["micro_recall"]) == (0.75, 0.75)
