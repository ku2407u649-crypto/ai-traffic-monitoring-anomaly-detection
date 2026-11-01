"""Evaluate anomaly detection: compare events.csv with a ground-truth CSV.

Ground truth CSV columns:  type,start_s,end_s   (one row per real incident)

  python evaluate_events.py --pred outputs/events.csv --truth ground_truth.csv --tolerance 3

An event counts as a true positive if its type matches and its time lies inside
[start_s - tolerance, end_s + tolerance]. Prints precision, recall and F1 per type and overall.
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tolerance", type=float, default=3.0)
    a = ap.parse_args()
    P, T = pd.read_csv(a.pred), pd.read_csv(a.truth)
    rows, TP, FP, FN = [], 0, 0, 0
    for typ in sorted(set(P["type"]) | set(T["type"])):
        p, t = P[P["type"] == typ], T[T["type"] == typ]
        matched, tp = set(), 0
        for _, e in p.iterrows():
            hit = t[(t["start_s"] - a.tolerance <= e["t"]) & (e["t"] <= t["end_s"] + a.tolerance)]
            hit = [i for i in hit.index if i not in matched]
            if hit:
                matched.add(hit[0]); tp += 1
        fp, fn = len(p) - tp, len(t) - len(matched)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append([typ, tp, fp, fn, round(prec, 3), round(rec, 3), round(f1, 3)])
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rows.append(["OVERALL", TP, FP, FN, round(prec, 3), round(rec, 3), round(f1, 3)])
    print(pd.DataFrame(rows, columns=["type", "TP", "FP", "FN", "precision", "recall", "F1"]).to_string(index=False))


if __name__ == "__main__":
    main()
