from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv
from classifier import FEATURES, ClassifierError, classify_heart_disease, validate_patient

ROOT = Path(__file__).resolve().parent
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
    "oldpeak": ("Perubahan hasil ST saat olahraga", "Nilai oldpeak dari hasil tes; jangan diperkirakan."),
    "slope": ("Pola segmen ST saat puncak tes", "Pilih bentuk yang tercatat pada hasil tes."),
    "ca": ("Jumlah pembuluh utama yang terlihat", "Jumlah 0 sampai 3 dari pemeriksaan fluoroskopi."),
    "thal": ("Hasil pemeriksaan thal", "Pilih hasil yang tercantum pada laporan pemeriksaan."),
}
CHOICES = {"sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2], "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7]}
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
VALUE_UNITS = {
    "age": "tahun", "trestbps": "mmHg", "chol": "mg/dL",
    "thalach": "kali/menit", "oldpeak": "",
}


def format_patient_value(name, value):
    if name in CHOICE_LABELS:
        return CHOICE_LABELS[name].get(int(value), str(value))
    unit = VALUE_UNITS.get(name, "")
    number = f"{float(value):g}"
    return f"{number} {unit}".strip()

load_dotenv()
st.set_page_config(page_title="Pemeriksaan Risiko Penyakit Jantung - LLM", page_icon="❤️", layout="wide")
st.title("Pemeriksaan Risiko Penyakit Jantung - LLM")
st.caption("Praktikum klasifikasi edukasional. Hasil LLM bukan diagnosis atau saran medis.")

# Streamlit Cloud Secrets override local .env values. Locally, Secrets may not
# exist at all; in that case the classifier reads configuration from .env.
try:
    secret_values = dict(st.secrets)
except StreamlitSecretNotFoundError:
    secret_values = {}
for secret_name in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
    if secret_name in secret_values:
        os.environ[secret_name] = str(secret_values[secret_name])

st.info("Masukkan semua informasi yang diminta dari hasil pemeriksaan. Model memerlukan seluruh 13 data; usia dan jenis kelamin saja belum cukup untuk membuat prediksi.")
st.download_button("Unduh contoh berkas CSV", pd.read_csv(ROOT / "contoh_data_baru.csv").to_csv(index=False),
                   "contoh_data_baru.csv", "text/csv")
tab_form, tab_batch = st.tabs(["Isi formulir", "Unggah beberapa data"])
with tab_form:
    with st.form("patient"):
        values = {}
        defaults = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
        for name in ["age", "sex"]:
            label, help_text = FIELD_INFO[name]
            if name in CHOICES:
                values[name] = st.selectbox(label, CHOICES[name], index=CHOICES[name].index(defaults[name]),
                                            format_func=lambda value, key=name: CHOICE_LABELS[key][value], help=help_text)
            else:
                limits = {"age": (1, 120)}
                minimum, maximum = limits[name]
                values[name] = st.number_input(label, min_value=float(minimum), max_value=float(maximum),
                                               value=float(defaults[name]), step=0.1 if name == "oldpeak" else 1.0,
                                               help=help_text)
        with st.expander("Data dari pemeriksaan jantung dan laboratorium", expanded=True):
            st.caption("Jika tidak mengetahui suatu nilai, lihat hasil pemeriksaan terlebih dahulu. Jangan mengira-ngira.")
            for name in FEATURES[2:]:
                label, help_text = FIELD_INFO[name]
                if name in CHOICES:
                    values[name] = st.selectbox(label, CHOICES[name], index=CHOICES[name].index(defaults[name]),
                                                format_func=lambda value, key=name: CHOICE_LABELS[key][value], help=help_text)
                else:
                    limits = {"trestbps": (40, 300), "chol": (50, 700), "thalach": (40, 250), "oldpeak": (0, 10)}
                    minimum, maximum = limits[name]
                    values[name] = st.number_input(label, min_value=float(minimum), max_value=float(maximum),
                                                   value=float(defaults[name]), step=0.1 if name == "oldpeak" else 1.0,
                                                   help=help_text)
        submitted = st.form_submit_button("Jalankan klasifikasi")
    if submitted:
        try:
            with st.spinner("Meminta prediksi dari model..."):
                result = classify_heart_disease(values)
            result_label = "Tidak terindikasi penyakit jantung" if result["prediction"] == 0 else "Terindikasi penyakit jantung"
            class_code = result["prediction"]
            with st.container(border=True):
                result_col, class_col = st.columns([3, 1])
                with result_col:
                    st.caption("HASIL KLASIFIKASI LLM")
                    st.markdown(f"### {result_label}")
                with class_col:
                    st.metric("Kelas dataset", f"Kelas {class_code}")
                st.markdown("**Penjelasan model**")
                st.write(result["reason"])
                st.caption("Klasifikasi untuk data praktikum; bukan diagnosis atau saran medis.")
                st.divider()
                st.markdown("**Ringkasan data yang dimasukkan**")
                summary_fields = ["age", "sex", "cp", "trestbps", "chol", "thalach"]
                summary_cols = st.columns(3)
                for i, name in enumerate(summary_fields):
                    summary_cols[i % 3].metric(FIELD_INFO[name][0], format_patient_value(name, values[name]))
                with st.expander("Lihat seluruh 13 data input"):
                    detail_cols = st.columns(3)
                    for i, name in enumerate(FEATURES):
                        detail_cols[i % 3].markdown(
                            f"**{FIELD_INFO[name][0]}**  \n{format_patient_value(name, values[name])}"
                        )
                st.caption(f"Model: {result['model']} · Penyedia: {result['provider']}")
        except (ValueError, ClassifierError) as exc:
            st.error(str(exc))
with tab_batch:
    st.write("Untuk banyak baris sekaligus, unggah CSV dengan 13 kolom fitur. Gunakan contoh berkas untuk melihat format kolom.")
    upload = st.file_uploader("Pilih berkas CSV", type="csv")
    if upload:
        try:
            frame = pd.read_csv(upload)
            if list(frame.columns) != FEATURES:
                raise ValueError(f"Kolom harus tepat dan berurutan: {', '.join(FEATURES)}")
            if len(frame) > 30:
                raise ValueError("Maksimal 30 baris per batch sesuai instruksi praktikum.")
            if len(frame) == 0:
                raise ValueError("CSV tidak berisi baris data.")
            # Validate every row before making any network requests.
            clean = [validate_patient(row) for row in frame.to_dict(orient="records")]
            if st.button("Jalankan prediksi batch"):
                results, failures = [], []
                progress = st.progress(0)
                for index, patient in enumerate(clean):
                    started = time.perf_counter()
                    try:
                        prediction = classify_heart_disease(patient)
                        label_id = "Tidak terindikasi penyakit jantung" if prediction["prediction"] == 0 else "Terindikasi penyakit jantung"
                        results.append({**patient, "prediction": prediction["prediction"],
                                        "label": label_id, "reason": prediction["reason"],
                                        "latency_seconds": round(time.perf_counter() - started, 3), "error": ""})
                    except ClassifierError as exc:
                        failures.append(index + 1)
                        results.append({**patient, "prediction": None, "label": None, "reason": "",
                                        "latency_seconds": round(time.perf_counter() - started, 3), "error": str(exc)})
                    progress.progress((index + 1) / len(clean))
                    if index + 1 < len(clean):
                        time.sleep(3)
                output = pd.DataFrame(results)
                output_id = output.rename(columns={"age": "usia", "sex": "jenis_kelamin", "cp": "jenis_nyeri_dada", "trestbps": "tekanan_darah_istirahat", "chol": "kolesterol", "fbs": "gula_darah_puasa_di_atas_120", "restecg": "hasil_ekg_istirahat", "thalach": "detak_jantung_maksimum", "exang": "nyeri_dada_saat_olahraga", "oldpeak": "perubahan_st_saat_olahraga", "slope": "pola_segmen_st", "ca": "jumlah_pembuluh_utama", "thal": "hasil_thal", "prediction": "prediksi_kode", "label": "hasil_prediksi", "reason": "penjelasan_model", "latency_seconds": "waktu_prediksi_detik", "error": "pesan_gagal"})
                st.dataframe(output_id, use_container_width=True)
                st.download_button("Unduh hasil CSV", output_id.to_csv(index=False), "hasil_prediksi.csv", "text/csv")
                st.info(f"Berhasil: {len(clean)-len(failures)} · Gagal: {len(failures)} · Tidak ada prediksi tebakan.")
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"CSV tidak valid: {exc}")
