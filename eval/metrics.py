"""Stdlib-only precision / recall maths."""
from collections import Counter, defaultdict


def precision_recall(y_true, y_pred, labels):
    per = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}

    def macro(key):
        vals = [v[key] for v in per.values() if v[key] is not None]
        return sum(vals) / len(vals) if vals else None

    n = len(y_true)
    return {
        "per_label": per,
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else None,
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "n": n,
    }


def multilabel_pr(true_sets, pred_sets, labels):
    """Per-category precision / recall when each CV can have several categories (primary + secondary).
    A category counts as predicted / expected for a CV if it is in that CV's set. Also returns micro averages."""
    per, TP, FP, FN = {}, 0, 0, 0
    for lab in labels:
        tp = sum(1 for t, p in zip(true_sets, pred_sets) if lab in t and lab in p)
        fp = sum(1 for t, p in zip(true_sets, pred_sets) if lab not in t and lab in p)
        fn = sum(1 for t, p in zip(true_sets, pred_sets) if lab in t and lab not in p)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        per[lab] = {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                    "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return {"per_label": per, "micro_precision": TP / (TP + FP) if TP + FP else None,
            "micro_recall": TP / (TP + FN) if TP + FN else None}


def confusion(y_true, y_pred):
    m = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return {t: dict(c) for t, c in m.items()}
