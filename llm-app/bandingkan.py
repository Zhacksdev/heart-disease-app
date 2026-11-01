"""Bandingkan DNN (praktikum-1b) dan LLM (praktikum-2b) pada BARIS TEST YANG SAMA.
Jalankan di folder praktikum-2b setelah train_model.py dan evaluate.py selesai:
    python bandingkan.py
    python bandingkan.py <test_predictions.csv DNN> <test_predictions_llm.csv>
Menghasilkan evaluation_results/perbandingan.md dan perbandingan.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

dnn_path = Path(sys.argv[1] if len(sys.argv) > 1 else "../praktikum-1b/artifacts/test_predictions.csv")
llm_path = Path(sys.argv[2] if len(sys.argv) > 2 else "evaluation_results/test_predictions_llm.csv")
dnn, llm = pd.read_csv(dnn_path), pd.read_csv(llm_path)
llm_ok = llm.dropna(subset=["prediction"])
common = dnn.merge(llm_ok[["row_index", "prediction", "latency_seconds"]], on="row_index", suffixes=("_dnn", "_llm"))


def ringkas(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {"n": int(len(y)), "accuracy": round(accuracy_score(y, p), 4), "precision": round(pr, 4),
            "recall": round(rc, 4), "f1_positif": round(f1, 4),
            "f1_macro": round(f1_score(y, p, average="macro", zero_division=0), 4),
            "FN": int(cm[1][0]), "FP": int(cm[0][1]), "confusion_matrix": cm.tolist()}


hasil = {
    "DNN (semua baris test)": ringkas(dnn["actual"], dnn["prediction"]),
    "DNN (baris yang sama dgn LLM)": ringkas(common["actual"], common["prediction_dnn"]),
    "LLM (baris yang dievaluasi)": ringkas(common["actual"], common["prediction_llm"].astype(int)),
}
hasil["_info"] = {"baris_test_total": int(len(dnn)), "baris_dibandingkan": int(len(common)),
                  "latensi_rata2_llm_detik": round(float(common["latency_seconds"].mean()), 3),
                  "sama_prediksi": int((common["prediction_dnn"] == common["prediction_llm"]).sum())}
Path("evaluation_results").mkdir(exist_ok=True)
Path("evaluation_results/perbandingan.json").write_text(json.dumps(hasil, indent=2), encoding="utf-8")
cols = ["n", "accuracy", "precision", "recall", "f1_positif", "f1_macro", "FN", "FP"]
md = "| Model | " + " | ".join(cols) + " | confusion [[TN,FP],[FN,TP]] |\n|" + "---|" * (len(cols) + 2) + "\n"
for k, v in hasil.items():
    if k.startswith("_"):
        continue
    md += f"| {k} | " + " | ".join(str(v[c]) for c in cols) + f" | {v['confusion_matrix']} |\n"
md += f"\nBaris dibandingkan: {hasil['_info']['baris_dibandingkan']} dari {hasil['_info']['baris_test_total']}; " \
      f"prediksi sama: {hasil['_info']['sama_prediksi']}; latensi rata-rata LLM: {hasil['_info']['latensi_rata2_llm_detik']} detik\n"
Path("evaluation_results/perbandingan.md").write_text(md, encoding="utf-8")
print(md)
