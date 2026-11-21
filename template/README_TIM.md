# [BearingGuard AI] · Tim [duoSumatera]

> Salin berkas ini menjadi `README.md` di repositori tim kalian, lalu isi semua bagian di dalam kurung siku.

**Kasus:** [4 diagnosis bearing] · **Aplikasi:** [(https://project-ml-diagnosis-kerusakan-bearing-happy-jia.streamlit.app/#bearing-guard-ai)] · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| [Nurjiha] | [2400515] | [EDA, preprocessing awal, penyusunan laporan, dan editing video] |
| [Raysha Happy] | [2404021] | [Penyelesaian notebook Colab, pengembangan model ML, evaluasi, tuning, penyimpanan model, aplikasi Streamlit, integrasi, dan GitHub] |

## Masalah dan pengguna

[BearingGuard AI dikembangkan untuk membantu teknisi pemeliharaan di pabrik semen dalam mengidentifikasi kondisi bearing berdasarkan data pengukuran getaran mesin. Sistem ini membantu mempercepat proses diagnosis dengan mengklasifikasikan kondisi bearing menjadi normal, outer_race, inner_race, dan bola, serta menampilkan probabilitas prediksi dan rekomendasi suku cadang yang perlu disiapkan sebagai bahan pertimbangan tindakan pemeliharaan.]

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| [Missing value pada suhu housing C] | [153] | [Baris dengan nilai kosong dihapus] |
| [Missing value pada kurtosis] | [456] | [Baris dengan nilai kosong dihapus] |
| [Label kondisi tidak konsisten] | [16] | [Dipetakan menjadi 4 kelas standar] |
| [Kolom id rekaman tidak relevan] | [1] | [Dihapus dari kolom model] |
| [Posisi sensor berbentuk kategorikal] | [2] | [Encoding: drive_end = 0, fan_end = 1] |


## Hasil

| Model | [Validation Macro F1] | [Test Macro F] |
|---|---|---|
| *Baseline*: [Logistic Regression] | [0.6120] | [0.6045] |
| [Tuned Extra Trees Classifier] | [0.8029] | [0.7992] |

*Threshold* yang dipakai: Tidak menggunakan threshold khusus (default). Model menentukan kelas hasil prediksi berdasarkan probabilitas tertinggi (argmax). Pendekatan ini digunakan karena sistem melakukan klasifikasi multikelas terhadap empat kondisi bearing, yaitu normal, outer_race, inner_race, dan bola.

## Cara menjalankan

```bash
1. Clone Repository
git clone https://github.com/rayshadf01-cyber/Project-ML-Diagnosis-Kerusakan-Bearing.git

2. Masuk ke Folder Proyek
cd Project-ML-Diagnosis-Kerusakan-Bearing

3. Install Dependencies
pip install -r requirements.txt

4. Jalankan Aplikasi
streamlit run app.py

Setelah aplikasi berhasil dijalankan, buka alamat lokal yang ditampilkan di terminal melalui browser untuk menggunakan BearingGuard AI.
```

## Struktur repositori

```
bearing-project/
app.py                  # Aplikasi web Streamlit
requirements.txt        # Daftar library yang digunakan
README.md               # Dokumentasi proyek
models/
├── model.joblib        # Model Tuned Extra Trees
└── metadata.json       # Metadata model dan fitur
src/
└── features.py         # Fungsi preprocessing dan
                        # transformasi fitur
```

## Keterbatasan

[1. Ketergantungan pada kualitas sensor: Hasil prediksi model bergantung pada akurasi dan kalibrasi sensor getaran serta sensor suhu. Data sensor yang tidak akurat dapat memengaruhi hasil diagnosis.

2. Penggunaan fitur hasil ekstraksi: Model menggunakan fitur statistik dan frekuensi dari rekaman getaran berdurasi satu detik, bukan sinyal getaran mentah (raw time-series waveform). Oleh karena itu, kerusakan bearing pada tahap awal yang belum menunjukkan perubahan getaran secara signifikan mungkin sulit terdeteksi.

3. Kesalahan klasifikasi: Model masih berpotensi mengalami kesalahan dalam membedakan jenis kerusakan outer_race, inner_race, dan bola karena adanya kemiripan karakteristik energi frekuensi getaran.

4. Keterbatasan cakupan dataset: Model dikembangkan berdasarkan dataset yang tersedia. Kemampuannya dalam mengenali kerusakan pada berbagai jenis mesin, kondisi operasional, dan lingkungan pabrik yang berbeda masih perlu diuji lebih lanjut.

5. Perlu validasi teknis: Hasil prediksi merupakan alat bantu pengambilan keputusan (decision-support tool), bukan pengganti pemeriksaan fisik. Keputusan pemeliharaan dan penggantian komponen tetap perlu dikonfirmasi melalui inspeksi teknis oleh tenaga ahli.]

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [ ] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [ ] *Baseline* wajib dilaporkan dan dibandingkan
- [ ] Data *test* hanya dibuka satu kali, di akhir
- [ ] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [ ] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
