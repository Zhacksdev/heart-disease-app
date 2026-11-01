"""Evaluate the LLM on a fixed stratified 80:20 split (seed 42).

Perubahan dari versi awal:
- Baris test yang punya nilai kosong (ca/thal) DILEWATI dan dicatat (LLM tidak boleh menebak nilai kosong),
  bukan menghentikan seluruh evaluasi.
- Prediksi per baris disimpan ke evaluation_results/test_predictions_llm.csv agar bisa dibandingkan
  dengan DNN pada baris yang sama (lihat bandingkan.py).
- Batas output API 128 token dan retry maksimal 3 kali untuk rate limit sementara.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from classifier import FEATURES, ClassifierError, classify_heart_disease

DATA_PATH = Path("data/heart_disease.csv")
OUT = Path("evaluation_results")
SEED = 42


def main() -> None:
    data = pd.read_csv(DATA_PATH, na_values="?")
    required = FEATURES + ["num"]
    if list(data.columns) != required:
        raise ValueError(f"Kolom dataset harus: {required}")
    data["target"] = (pd.to_numeric(data["num"], errors="raise") > 0).astype(int)
    missing = data[FEATURES + ["num"]].isna().sum().to_dict()
    train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=SEED, stratify=data["target"])
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({"row_index": test_idx}).to_csv(OUT / "test_indices_seed42.csv", index=False)

    complete = data.loc[test_idx, FEATURES].notna().all(axis=1)
    skipped = [int(i) for i in test_idx[~complete.values]]
    eval_idx = list(test_idx[complete.values])
    print("Rows:", len(data), "missing:", {k: v for k, v in missing.items() if v}, "class distribution:", data.target.value_counts().sort_index().to_dict())
    print(f"Train/test: {len(train_idx)}/{len(test_idx)}; dievaluasi: {len(eval_idx)}; dilewati (nilai kosong): {len(skipped)} -> {skipped}")

    rows, failures, latencies, invalid_json, retries_total = [], [], [], 0, 0
    for position, idx in enumerate(eval_idx):
        record = data.loc[idx, FEATURES].to_dict()  # Labels never enter the request.
        started = time.perf_counter()
        actual = int(data.loc[idx, "target"])
        try:
            answer = classify_heart_disease(record)
            latency = time.perf_counter() - started
            latencies.append(latency)
            retries_total += answer.get("api_retries", 0)
            rows.append({"row_index": int(idx), "actual": actual, "prediction": answer["prediction"],
                         "latency_seconds": round(latency, 3), "api_retries": answer.get("api_retries", 0),
                         "reason": answer["reason"], "error": ""})
        except ClassifierError as exc:
            retries_total += exc.retries
            failures.append({"row_index": int(idx), "error": str(exc), "api_retries": exc.retries})
            if "JSON" in str(exc):
                invalid_json += 1
            rows.append({"row_index": int(idx), "actual": actual, "prediction": None,
                         "latency_seconds": round(time.perf_counter() - started, 3), "api_retries": exc.retries,
                         "reason": "", "error": str(exc)})
        print(f"{position+1}/{len(eval_idx)}", flush=True)
        if position + 1 < len(eval_idx):
            # Slow requests to fit the low OTPM limits on the Groq on-demand tier.
            time.sleep(3)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "test_predictions_llm.csv", index=False)
    ok = res.dropna(subset=["prediction"])
    y_true, y_pred = ok["actual"].astype(int).tolist(), ok["prediction"].astype(int).tolist()
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": os.getenv("LLM_PROVIDER", "groq"),
        "model": os.getenv("LLM_MODEL") or "provider default",
        "temperature": 0, "seed": SEED, "test_size": 0.2,
        "test_rows_total": int(len(test_idx)), "test_rows_skipped_missing": len(skipped),
        "skipped_row_indices": skipped, "test_rows_sent": len(eval_idx),
        "requests_succeeded": len(y_pred), "requests_failed": len(failures), "invalid_json": invalid_json,
        "rate_limit_retries_total": retries_total, "max_rate_limit_retries_per_request": 3,
        "coverage_of_total_test": round(len(y_pred) / len(test_idx), 4),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "accuracy": accuracy_score(y_true, y_pred) if y_pred else None,
        "precision_recall_f1_positive_class": [float(v) for v in precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[0:3]] if y_pred else None,
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if y_pred else None,
        "false_negative_count": int(sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["No Heart Disease", "Heart Disease"], output_dict=True, zero_division=0) if y_pred else None,
        "failures": failures,
        "note": "Metrics exclude skipped (missing value) and failed/invalid requests; always report coverage alongside metrics.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
