"""Streamlit UI for the trained Cleveland DNN."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
LABELS = {0: "Tidak terindikasi penyakit jantung", 1: "Terindikasi penyakit jantung"}
CHOICES = {"sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2], "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7]}
FIELD_INFO = {
    "age": ("Usia", "Umur pasien dalam tahun."),
    "sex": ("Jenis kelamin", "Pilih sesuai data pemeriksaan."),
    "cp": ("Jenis nyeri dada", "Jenis nyeri yang dicatat saat pemeriksaan."),
    "trestbps": ("Tekanan darah saat istirahat (mmHg)", "Gunakan angka sistolik dari hasil pemeriksaan."),
    "chol": ("Kolesterol (mg/dL)", "Nilai kolesterol serum dari hasil laboratorium."),
    "fbs": ("Gula darah puasa di atas 120 mg/dL?", "Pilih berdasarkan hasil pemeriksaan gula darah puasa."),
    "restecg": ("Hasil EKG saat istirahat", "Pilih keterangan yang tercantum pada hasil EKG."),
    "thalach": ("Detak jantung maksimum saat tes", "Nilai maksimum yang tercatat saat pemeriksaan/tes olahraga."),
    "exang": ("Nyeri dada saat olahraga/tes?", "Pilih Ya jika tercatat muncul saat tes aktivitas."),
    "oldpeak": ("Perubahan hasil ST saat olahraga", "Nilai oldpeak dari hasil tes; ini bukan nilai yang perlu diperkirakan."),
    "slope": ("Pola segmen ST saat puncak tes", "Pilih bentuk yang tercatat pada hasil tes."),
    "ca": ("Jumlah pembuluh utama yang terlihat", "Jumlah 0 sampai 3 dari pemeriksaan fluoroskopi."),
    "thal": ("Hasil pemeriksaan thal", "Pilih hasil yang tercantum pada laporan pemeriksaan."),
}
CHOICE_LABELS = {
    "sex": {0: "Perempuan", 1: "Laki-laki"},
    "cp": {1: "Nyeri dada khas", 2: "Nyeri dada tidak khas", 3: "Nyeri bukan angina", 4: "Tanpa gejala nyeri dada"},
    "fbs": {0: "Tidak", 1: "Ya"},
    "restecg": {0: "Normal", 1: "Perubahan gelombang ST-T", 2: "Pembesaran bilik jantung"},
    "exang": {0: "Tidak", 1: "Ya"},
    "slope": {1: "Menanjak", 2: "Datar", 3: "Menurun"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Kelainan tetap", 7: "Kelainan yang dapat berbalik"},
}

st.set_page_config(page_title="Pemeriksaan Risiko Jantung - DNN", page_icon="❤️", layout="wide")
st.title("Pemeriksaan Risiko Penyakit Jantung - DNN")
st.caption("Alat praktikum machine learning. Hasil model bukan diagnosis atau saran medis.")


@st.cache_resource
def load_artifacts():
    model_file, preprocessor_file = ARTIFACTS / "dnn_model.joblib", ARTIFACTS / "preprocessor.joblib"
    if not model_file.exists() or not preprocessor_file.exists():
        return None, None
    return joblib.load(model_file), joblib.load(preprocessor_file)


model, preprocessor = load_artifacts()
if model is None:
    st.warning("Model belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
else:
    sample_path = ROOT / "contoh_data_baru.csv"
    st.info("Masukkan semua informasi yang diminta dari hasil pemeriksaan. Model memerlukan seluruh 13 data; usia dan jenis kelamin saja belum cukup untuk membuat prediksi.")
    st.download_button("Unduh contoh berkas CSV", sample_path.read_bytes(), "contoh_data_baru.csv", "text/csv")
    form_tab, batch_tab = st.tabs(["Isi formulir", "Unggah beberapa data"])
    with form_tab:
        with st.form("patient_form"):
            patient = {}
            defaults = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
            for name in ["age", "sex"]:
                label, help_text = FIELD_INFO[name]
                patient[name] = st.selectbox(label, CHOICES[name], index=CHOICES[name].index(defaults[name]), format_func=lambda value, key=name: CHOICE_LABELS[key][value], help=help_text) if name in CHOICES else st.number_input(label, min_value=1, max_value=120, value=defaults[name], step=1, help=help_text)
            with st.expander("Data dari pemeriksaan jantung dan laboratorium", expanded=True):
                st.caption("Jika tidak mengetahui suatu nilai, lihat hasil pemeriksaan terlebih dahulu. Jangan mengira-ngira.")
                for name in FEATURES[2:]:
                    label, help_text = FIELD_INFO[name]
                    if name in CHOICES:
                        patient[name] = st.selectbox(label, CHOICES[name], index=CHOICES[name].index(defaults[name]), format_func=lambda value, key=name: CHOICE_LABELS[key][value], help=help_text)
                    else:
                        limits = {"trestbps": (40, 300), "chol": (50, 700), "thalach": (40, 250), "oldpeak": (0, 10)}
                        minimum, maximum = limits[name]
                        patient[name] = st.number_input(label, min_value=float(minimum), max_value=float(maximum), value=float(defaults[name]), step=0.1 if name == "oldpeak" else 1.0, help=help_text)
            submitted = st.form_submit_button("Jalankan klasifikasi")
        if submitted:
            try:
                row = pd.DataFrame([patient], columns=FEATURES)
                probability = float(model.predict_proba(preprocessor.transform(row))[0, list(model.classes_).index(1)])
                label = int(probability >= 0.5)
                st.success(f"Hasil klasifikasi model: {LABELS[label]}")
                st.metric("Skor prediksi kelas penyakit jantung", f"{probability:.1%}")
                st.caption("Skor ini menunjukkan keluaran model pada data praktikum, bukan peluang diagnosis klinis.")
            except Exception as exc:
                st.error(f"Input gagal diproses: {exc}")
    with batch_tab:
        st.write("Untuk banyak baris sekaligus, unggah CSV dengan 13 kolom fitur. Gunakan contoh berkas untuk melihat format kolom.")
        uploaded = st.file_uploader("Pilih berkas CSV", type="csv")
        if uploaded:
            try:
                frame = pd.read_csv(uploaded)
                if list(frame.columns) != FEATURES:
                    raise ValueError(f"CSV wajib berisi 13 kolom fitur dalam urutan ini: {', '.join(FEATURES)}")
                if frame.empty:
                    raise ValueError("CSV tidak memiliki baris data.")
                if st.button("Jalankan prediksi batch"):
                    transformed = preprocessor.transform(frame)
                    probabilities = model.predict_proba(transformed)[:, list(model.classes_).index(1)]
                    result = frame.copy()
                    result["probability_heart_disease"] = probabilities
                    result["prediction"] = (probabilities >= 0.5).astype(int)
                    result["label"] = result["prediction"].map(LABELS)
                    result_id = result.rename(columns={"age": "usia", "sex": "jenis_kelamin", "cp": "jenis_nyeri_dada", "trestbps": "tekanan_darah_istirahat", "chol": "kolesterol", "fbs": "gula_darah_puasa_di_atas_120", "restecg": "hasil_ekg_istirahat", "thalach": "detak_jantung_maksimum", "exang": "nyeri_dada_saat_olahraga", "oldpeak": "perubahan_st_saat_olahraga", "slope": "pola_segmen_st", "ca": "jumlah_pembuluh_utama", "thal": "hasil_thal", "probability_heart_disease": "skor_penyakit_jantung", "prediction": "prediksi_kode", "label": "hasil_prediksi"})
                    st.dataframe(result_id, use_container_width=True)
                    st.download_button("Unduh hasil CSV", result_id.to_csv(index=False), "hasil_prediksi.csv", "text/csv")
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"CSV tidak valid: {exc}")

with st.expander("Informasi model dan evaluasi (untuk praktikum)"):
    info_file, metrics_file = ARTIFACTS / "model_info.json", ARTIFACTS / "metrics.json"
    if info_file.exists():
        st.json(json.loads(info_file.read_text(encoding="utf-8")))
    if metrics_file.exists():
        st.json(json.loads(metrics_file.read_text(encoding="utf-8")))
