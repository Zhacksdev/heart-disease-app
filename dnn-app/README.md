# DNN Cleveland Heart Disease — Praktikum 1B

Aplikasi Streamlit untuk klasifikasi edukasional menggunakan DNN kecil pada dataset Cleveland. Hasil bukan diagnosis medis.

## Menjalankan

```bash
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Training menghasilkan `artifacts/dnn_model.joblib`, `artifacts/preprocessor.joblib`, `artifacts/model_info.json`, `artifacts/metrics.json`, indeks test seed 42, dan prediksi test. Artefak perlu ikut disertakan saat aplikasi dideploy.

Model menggunakan MLPClassifier scikit-learn dengan hidden layers 32 dan 16, ReLU, Adam, early stopping (validation 20% dari data training), batch size 32, maksimum 500 iterasi, dan threshold 0.5. Kategori di-one-hot encode; fitur numerik diimputasi median dan distandardisasi; kategori diimputasi modus. Preprocessor hanya di-fit pada training set. Tidak menggunakan class weights.

Split stratified 80:20 random seed 42 sejalan dengan proyek LLM praktikum 2B. Hasil ditulis ke `artifacts/metrics.json`. Evaluasi mencakup accuracy, precision, recall, F1, confusion matrix, classification report, dan false negative. Karena dataset kecil, hasil satu split memiliki variasi; false negative perlu diperhatikan dan accuracy saja tidak cukup.

Aplikasi menyediakan form pasien, upload CSV berfitur saja, hasil prediksi dan probabilitas untuk diunduh. Contoh input tersedia di `contoh_data_baru.csv`. Untuk deployment, salin seluruh folder `artifacts` setelah menjalankan training. Jangan unggah data pasien pribadi.
