# Bearing Fault Diagnosis

## Deskripsi Proyek
Proyek Machine Learning untuk mengklasifikasikan kondisi bearing berdasarkan data getaran mesin.

## Kelas Prediksi
- normal
- outer_race
- inner_race
- bola

## Model
Tuned Extra Trees Classifier.

## Evaluasi
- Validation Macro F1: 0.8029
- Test Macro F1: 0.7992
- Test Accuracy: 0.8427

## Fitur Input
- posisi_sensor
- kecepatan_rpm
- beban_kN
- suhu_housing_C
- rms_g
- puncak_g
- kurtosis
- crest_factor
- energi_bpfo
- energi_bpfi
- energi_bsf
- frekuensi_dominan_hz

## Menjalankan Aplikasi

Install dependencies:

pip install -r requirements.txt

Jalankan Streamlit:

streamlit run app.py

## Struktur Proyek
- app.py
- requirements.txt
- src/features.py
- models/model.joblib
- models/metadata.json
- README.md

## Catatan
Hasil prediksi merupakan alat bantu analisis dan perlu dikonfirmasi melalui inspeksi teknis.
