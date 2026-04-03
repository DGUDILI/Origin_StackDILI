"""
비교표 생성: env1/env2 × original/clean 결과를 하나의 표로 합침
"""
import os
import json
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DGUDILI = os.path.join(ROOT, "DGUDILI_2026")

COLS = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]


def load_env1(suffix=""):
    """outputs[_clean]/results_comparison_tune_cls.csv → StackDILI + DGUDILI 행"""
    csv = os.path.join(DGUDILI, f"outputs{suffix}", "results_comparison_tune_cls.csv")
    if not os.path.exists(csv):
        print(f"[WARN] missing: {csv}")
        return {}
    df = pd.read_csv(csv, index_col=0)
    rows = {}
    for idx in df.index:
        rows[idx] = {c: df.loc[idx, c] for c in COLS if c in df.columns}
    return rows


def load_env2(suffix=""):
    """outputs_cv[_clean]/results_cv.csv → StackDILI(CV) + DGUDILI(CV) 행"""
    csv = os.path.join(DGUDILI, f"outputs_cv{suffix}", "results_cv.csv")
    if not os.path.exists(csv):
        print(f"[WARN] missing: {csv}")
        return {}
    df = pd.read_csv(csv, index_col=0)
    rows = {}
    for idx in df.index:
        rows[idx] = {c: df.loc[idx, c] for c in COLS if c in df.columns}
    return rows


def fmt(val):
    try:
        return f"{float(val):.4f}"
    except Exception:
        return str(val)


def print_table(rows: dict, title: str):
    if not rows:
        print(f"\n[{title}] 결과 없음\n")
        return
    header = f"{'Model':<32}" + "".join(f"{c:>11}" for c in COLS)
    sep    = "-" * (32 + 11 * len(COLS))
    print(f"\n{'=' * len(sep)}")
    print(title)
    print("=" * len(sep))
    print(header)
    print(sep)
    for name, metrics in rows.items():
        row = f"{name:<32}" + "".join(f"{fmt(metrics.get(c, 'N/A')):>11}" for c in COLS)
        print(row)
    print("=" * len(sep))


def main():
    print("\n" + "=" * 80)
    print("DGUDILI 2026  전체 비교표")
    print("=" * 80)

    # ── env1 (fixed split) ──────────────────────────────────────────────────
    e1_orig  = load_env1("")
    e1_clean = load_env1("_clean")

    print_table(e1_orig,  "env1 | Fixed Split | Original Data")
    print_table(e1_clean, "env1 | Fixed Split | Clean Data")

    # ── env2 (10-Fold CV) ───────────────────────────────────────────────────
    e2_orig  = load_env2("")
    e2_clean = load_env2("_clean")

    print_table(e2_orig,  "env2 | 10-Fold CV  | Original Data")
    print_table(e2_clean, "env2 | 10-Fold CV  | Clean Data")

    # ── 통합 CSV 저장 ────────────────────────────────────────────────────────
    all_rows = []
    for label, rows in [
        ("env1_orig",  e1_orig),
        ("env1_clean", e1_clean),
        ("env2_orig",  e2_orig),
        ("env2_clean", e2_clean),
    ]:
        for model, metrics in rows.items():
            all_rows.append({"env": label, "model": model, **metrics})

    if all_rows:
        out_csv = os.path.join(ROOT, "comparison_all.csv")
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\n통합 CSV 저장: {out_csv}")
    else:
        print("\n[WARN] 저장할 결과가 없습니다. 파이프라인을 먼저 실행하세요.")


if __name__ == "__main__":
    main()
