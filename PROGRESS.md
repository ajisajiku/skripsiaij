# Laporan Progres Proyek: Prediksi Diabetes dengan GWO-Optimized Decision Tree

## Tujuan Proyek
Membuat program untuk memprediksi diabetes menggunakan model Decision Tree yang hyperparameter-nya dioptimasi dengan Grey Wolf Optimizer (GWO), kemudian membungkusnya dalam aplikasi web Streamlit.

## Progres Tahap Demi Tahap

### Tahap 1: Preprocessing Data & Model Baseline
**Tujuan:** Membuat pipeline preprocessing data yang solid dan melatih model Decision Tree dasar sebagai baseline.

**Langkah-langkah yang Diimplementasikan:**
1.  **Pemuatan Data:** Memuat dataset `diabetes.csv` ke dalam DataFrame.
2.  **Penanganan Nilai 0:** Mengganti nilai 0 di kolom-kolom kritis (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) dengan `np.nan`.
3.  **Pemisahan Fitur & Target:** Memisahkan `X` (fitur) dan `y` (target 'Outcome').
4.  **Pembagian Data:** Membagi data menjadi `X_train`, `X_test`, `y_train`, `y_test` (`test_size=0.2`, `random_state=42`).
5.  **Imputasi:** Menggunakan `SimpleImputer(strategy='median')` yang di-fit HANYA pada `X_train`, lalu mentransformasi `X_train` dan `X_test`.
6.  **Penskalaan Fitur:** Menggunakan `StandardScaler()` yang di-fit HANYA pada `X_train`, lalu mentransformasi `X_train` dan `X_test`.
7.  **Model Baseline:** Melatih `DecisionTreeClassifier` dengan parameter default (`random_state=42`).

**Metrik Baseline Final (pada data uji yang tidak seimbang):**
-   Accuracy: 0.7273
-   Precision: 0.6102
-   **Recall: 0.6545**
-   F1-Score: 0.6316
-   ROC-AUC Score: 0.7111

**Kesimpulan Tahap 1:** Pipeline preprocessing yang metodologinya benar telah ditetapkan, dan model baseline telah dievaluasi.

### Tahap 2: Optimasi GWO & Perbandingan Model
**Tujuan:** Mengoptimalkan hyperparameter Decision Tree menggunakan GWO dan membandingkannya dengan metode lain.

**Langkah-langkah yang Diimplementasikan:**
1.  **Fungsi Fitness (`calculate_fitness`):** Dibuat untuk mengevaluasi model Decision Tree menggunakan 5-fold cross-validation dengan `scoring='recall'`, dan mengembalikan `1.0 - avg_recall` (untuk minimisasi GWO).
2.  **Algoritma GWO (`run_gwo`):** Diimplementasikan dari awal, menerima fungsi fitness, batas pencarian, dimensi, jumlah serigala, dan jumlah iterasi.
3.  **Ruang Pencarian Hyperparameter:**
    *   `max_depth`: [1, 20]
    *   `min_samples_leaf`: [1, 50]
4.  **Parameter GWO:** `num_wolves=10`, `max_iterations=20`.
5.  **Hasil Optimasi GWO (dari verifikasi terakhir):**
    *   Parameter Terbaik: `max_depth=4`, `min_samples_leaf=21`, `criterion='gini'`.
    *   Recall CV Terbaik: `0.7462`.
6.  **Evaluasi Model GWO (pada data uji):**
    *   Accuracy: 0.7013
    *   Precision: 0.5600
    *   **Recall: 0.7636**
    *   F1-Score: 0.6462
    *   ROC-AUC Score: 0.8028
7.  **Perbandingan dengan RandomizedSearchCV:**
    *   Parameter Terbaik: `max_depth=4`, `min_samples_leaf=30`, `criterion='entropy'`.
    *   Recall: `0.6909`.

**Kesimpulan Tahap 2:** Model GWO berhasil meningkatkan Recall secara signifikan (dari 65.5% menjadi 76.4%), yang merupakan tujuan utama. Meskipun akurasi sedikit menurun, peningkatan Recall dan ROC-AUC menjadikan model GWO sebagai pilihan terbaik untuk kasus ini.

### Tahap 3: Pengembangan Aplikasi Streamlit (`app.py`)
**Tujuan:** Membangun antarmuka web interaktif untuk demonstrasi dan prediksi.

**Struktur Aplikasi (`app.py`):**
-   **Konfigurasi Halaman:** `page_title='Optimasi GWO-DT'`, `layout='wide'`.
-   **Navigasi Sidebar:** Menggunakan `st.sidebar.radio` untuk 4 modul:
    1.  `1. Upload & Preprocessing`
    2.  `2. Optimasi Model`
    3.  `3. Evaluasi & Perbandingan`
    4.  `4. Prediksi Interaktif`
-   **Penyimpanan Data:** Menggunakan `st.session_state` untuk menyimpan data yang telah diproses (`X_train_scaled`, `y_train`, `X_test_scaled`, `y_test`), objek preprocessing (`scaler`, `imputer`), nama fitur (`feature_names`), dan model-model terlatih (`base_model`, `gwo_model`, `rand_model`).

**Modul yang Sudah Diimplementasikan di `app.py`:**
1.  **Modul 1: Upload & Preprocessing Data**
    *   Memungkinkan pengguna mengunggah file CSV.
    *   Menampilkan 5 baris pertama data.
    *   Tombol untuk memulai pipeline preprocessing (mengganti 0 ke NaN, split, imputasi, scaling).
    *   Menyimpan semua hasil preprocessing ke `st.session_state`.
2.  **Modul 2: Optimasi Model**
    *   Memeriksa ketersediaan data yang sudah diproses.
    *   Tombol untuk memulai proses optimasi.
    *   Melatih 3 model secara berurutan:
        *   Model Baseline (default Decision Tree).
        *   Model GWO (memanggil fungsi `run_gwo`).
        *   Model RandomizedSearchCV.
    *   Menyimpan ketiga model terlatih dan parameter terbaiknya ke `st.session_state`.
3.  **Modul 3: Evaluasi & Perbandingan Model**
    *   Memeriksa ketersediaan model yang sudah dilatih.
    *   Mengambil model dan data uji dari `st.session_state`.
    *   Melakukan prediksi untuk setiap model.
    *   Menghitung dan menampilkan tabel perbandingan metrik (`Recall`, `Precision`, `F1-Score`, `Accuracy`) untuk ketiga model.

**Modul yang Sudah Diimplementasikan di `app.py`:**
1.  **Modul 1: Upload & Preprocessing Data**
    *   Memungkinkan pengguna mengunggah file CSV.
    *   Menampilkan 5 baris pertama data.
    *   Tombol untuk memulai pipeline preprocessing (mengganti 0 ke NaN, split, imputasi, scaling).
    *   Menyimpan semua hasil preprocessing ke `st.session_state`.
2.  **Modul 2: Optimasi Model**
    *   Memeriksa ketersediaan data yang sudah diproses.
    *   Tombol untuk memulai proses optimasi.
    *   Melatih 3 model secara berurutan:
        *   Model Baseline (default Decision Tree).
        *   Model GWO (memanggil fungsi `run_gwo`).
        *   Model RandomizedSearchCV.
    *   Menyimpan ketiga model terlatih dan parameter terbaiknya ke `st.session_state`.
3.  **Modul 3: Evaluasi & Perbandingan Model**
    *   Memeriksa ketersediaan model yang sudah dilatih.
    *   Mengambil model dan data uji dari `st.session_state`.
    *   Melakukan prediksi untuk setiap model.
    *   Menghitung dan menampilkan tabel perbandingan metrik (`Recall`, `Precision`, `F1-Score`, `Accuracy`) untuk ketiga model.
    *   Menampilkan Confusion Matrix untuk ketiga model.
4.  **Modul 4: Prediksi Interaktif**
    *   Memungkinkan pengguna memasukkan 8 parameter pasien secara manual.
    *   Melakukan preprocessing pada input pengguna (mengganti 0 ke NaN, imputasi, scaling).
    *   Mendapatkan prediksi diabetes dari model GWO terbaik.
    *   Menampilkan hasil prediksi (Diabetes/Tidak Diabetes) beserta tingkat keyakinan.
    *   Menampilkan plot SHAP untuk menjelaskan faktor-faktor yang mempengaruhi prediksi.

**Status Saat Ini:** Semua modul di `app.py` (Modul 1, 2, 3, dan 4) sudah selesai diimplementasikan.

**Cara Menjalankan Aplikasi:**
Untuk menjalankan aplikasi Streamlit, buka terminal di direktori proyek dan jalankan:
```bash
python -m streamlit run app.py
```
Kemudian akses `http://localhost:8501` di browser Anda.
