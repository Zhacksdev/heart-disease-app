# Cleveland Heart Disease LLM classifier

Aplikasi praktikum Streamlit untuk membandingkan classifier LLM dengan DNN pada
dataset Cleveland. Prediksi hanya untuk pembelajaran dan bukan diagnosis medis.

## Menjalankan aplikasi

1. Buat virtual environment lalu pasang dependensi: `python -m pip install -r requirements.txt`.
2. Isi `.env` dengan `LLM_PROVIDER` (`groq` atau `gemini`), `LLM_API_KEY`, dan
   opsional `LLM_MODEL`. Default model: `llama-3.3-70b-versatile` (Groq) atau
   `gemini-2.0-flash` (Gemini).
3. Jalankan `python -m streamlit run app.py`. Cara ini tetap bekerja jika folder Scripts Python belum ditambahkan ke `PATH` PowerShell.

Untuk Streamlit Community Cloud, masukkan `LLM_PROVIDER`, `LLM_API_KEY`, dan
opsional `LLM_MODEL` pada pengaturan Secrets. File `.env` dan
`.streamlit/secrets.toml` diabaikan Git.

## Evaluasi

Jalankan `python evaluate.py`. Evaluasi memakai split stratified 80:20, seed 42,
dan fitur saja dalam request; target `num > 0` hanya dibaca di sisi evaluasi.
Indeks test ditulis ke `evaluation_results/test_indices_seed42.csv` supaya
implementasi DNN bisa menggunakan indeks yang sama. Evaluasi berhenti jika ada
missing value, sehingga LLM tidak menebak nilai yang kosong. Hasil metrik,
coverage, latency, dan kegagalan ditulis ke `evaluation_results/metrics.json`.
Output API dibatasi 128 token, jeda antarrequest 3 detik, dan rate limit sementara
dicoba ulang maksimal 3 kali mengikuti header waktu reset kuota. Request yang tetap
gagal atau JSON tidak valid tidak diubah menjadi label; metrik hanya mencakup
request sukses, jadi baca juga jumlah coverage dan kegagalan.

Input batch menerima maksimal 30 baris dan memvalidasi seluruh baris sebelum
mengirim request. Hasil menyediakan label, alasan, latency, dan kolom error.
`contoh_data_baru.csv` hanya berisi 13 fitur tanpa label.

## Kode kategori

`classifier.py` menyertakan codebook UCI di system prompt: sex, cp, fbs,
restecg, exang, slope, ca, dan thal. Seluruh fitur dicek sebelum dikirim.
