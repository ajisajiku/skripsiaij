# BAB 4: HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil implementasi sistem prediksi diabetes menggunakan Decision Tree yang dioptimasi dengan Grey Wolf Optimizer (GWO). Pembahasan mencakup hasil preprocessing data, pelatihan model, proses optimasi, evaluasi performa, dan perbandingan dengan metode lain.

## 4.1 Hasil Preprocessing Data

Dataset **Pima Indians Diabetes Database** yang digunakan dalam penelitian ini terdiri dari 768 sampel dengan 8 fitur prediktor dan 1 variabel target (Outcome). Tahap preprocessing dilakukan untuk memastikan kualitas data sebelum digunakan dalam pemodelan.

### 4.1.1 Identifikasi dan Penanganan Missing Values

Analisis awal menunjukkan bahwa beberapa kolom memiliki nilai 0 yang secara medis tidak masuk akal. Tabel 4.1 menunjukkan distribusi nilai 0 pada kolom-kolom tersebut.

**Tabel 4.1: Distribusi Nilai 0 (Invalid) pada Dataset**

| Kolom | Jumlah Nilai 0 | Persentase dari Total Data |
|-------|----------------|----------------------------|
| Glucose | 5 | 0.65% |
| BloodPressure | 35 | 4.56% |
| SkinThickness | 227 | 29.56% |
| Insulin | 374 | 48.70% |
| BMI | 11 | 1.43% |

Nilai 0 pada kolom-kolom tersebut dianggap sebagai data hilang (*missing values*) karena secara biologis tidak mungkin seseorang memiliki nilai Glucose atau BMI sama dengan nol. Oleh karena itu, nilai-nilai tersebut diganti dengan `NaN` dan kemudian diisi menggunakan strategi **imputasi median**.

Pemilihan median sebagai strategi imputasi didasarkan pada sifat median yang lebih robust terhadap outlier dibandingkan mean. Hal ini penting mengingat data medis sering mengandung nilai ekstrem yang bukan merupakan kesalahan pengukuran, melainkan kondisi pasien yang memang berbeda.

### 4.1.2 Pemisahan dan Standardisasi Data

Dataset dibagi menjadi dua bagian dengan rasio 80:20, yaitu:
- **Data Training**: 614 sampel (80%)
- **Data Testing**: 154 sampel (20%)

Pemisahan ini dilakukan dengan `random_state=42` untuk memastikan reprodusibilitas hasil. Data testing disimpan secara terpisah dan tidak digunakan sama sekali dalam proses training maupun optimasi hyperparameter untuk menghindari data leakage.

Setelah pemisahan, seluruh fitur numerik dinormalisasi menggunakan `StandardScaler` yang menghasilkan distribusi dengan mean 0 dan standar deviasi 1. Standardisasi ini penting untuk memastikan setiap fitur memiliki kontribusi yang seimbang dalam proses pemodelan.

---

## 4.2 Hasil Pelatihan Model Baseline

Model baseline Decision Tree dilatih menggunakan parameter default dari scikit-learn untuk menjadi acuan perbandingan terhadap model yang dioptimasi. Tabel 4.2 menampilkan karakteristik model baseline yang dihasilkan.

**Tabel 4.2: Karakteristik Model Baseline Decision Tree**

| Parameter | Nilai |
|-----------|-------|
| max_depth | None (tidak dibatasi) |
| min_samples_leaf | 1 |
| criterion | gini |
| Kedalaman Pohon Aktual | ~23 level |
| Jumlah Leaf Nodes | >200 nodes |
| Rata-rata Recall (5-Fold CV) | 0.6545 |

Model baseline menghasilkan pohon keputusan yang sangat dalam (23 level) dengan lebih dari 200 leaf nodes. Struktur pohon yang sangat kompleks ini mengindikasikan adanya **overfitting**, yaitu model terlalu menyesuaikan diri dengan data training sehingga kehilangan kemampuan generalisasi pada data baru.

### 4.2.1 Pembahasan Model Baseline

Visualisasi pohon keputusan baseline (dibatasi hingga kedalaman 3 untuk keterbacaan) menunjukkan bahwa model membuat banyak split yang sangat spesifik untuk data training. Hal ini menyebabkan model menghafal noise dalam data alih-alih mempelajari pola yang sesungguhnya.

Kompleksitas model baseline yang tinggi juga menyulitkan interpretasi oleh praktisi medis. Dalam aplikasi klinis, kemampuan untuk memahami alasan di balik keputusan model sangat penting untuk membangun kepercayaan dan memvalidasi prediksi secara medis.

---

## 4.3 Hasil Optimasi dengan RandomizedSearchCV

RandomizedSearchCV digunakan sebagai metode pembanding untuk optimasi hyperparameter Decision Tree. Metode ini melakukan pencarian acak dalam ruang parameter yang telah ditentukan.

### 4.3.1 Konfigurasi RandomizedSearchCV

**Tabel 4.3: Konfigurasi RandomizedSearchCV**

| Parameter | Nilai |
|-----------|-------|
| n_iter | 100 iterasi |
| cv | 5-fold cross-validation |
| scoring | recall |
| Search Space: criterion | ['gini', 'entropy'] |
| Search Space: max_depth | [1, 2, ..., 20] |
| Search Space: min_samples_leaf | [1, 2, ..., 50] |

### 4.3.2 Parameter Optimal RandomizedSearchCV

Setelah 100 iterasi random sampling, RandomizedSearchCV menemukan kombinasi hyperparameter terbaik yang ditampilkan pada Tabel 4.4.

**Tabel 4.4: Parameter Optimal dari RandomizedSearchCV**

| Parameter | Nilai Optimal |
|-----------|---------------|
| criterion | entropy |
| max_depth | 5 |
| min_samples_leaf | 18 |
| **Skor Recall (CV)** | **0.6909** |

Model RandomizedSearchCV menghasilkan pohon yang lebih sederhana dengan kedalaman maksimum 5 level dan minimal 18 sampel di setiap leaf. Hal ini menunjukkan bahwa RandomizedSearchCV berhasil menemukan parameter yang lebih baik dibanding baseline dalam hal mencegah overfitting.

### 4.3.3 Pembahasan Optimasi RandomizedSearchCV

RandomizedSearchCV membutuhkan waktu komputasi sekitar 5 menit untuk menyelesaikan 100 iterasi pada hardware yang digunakan. Metode ini efektif dalam mengeksplorasi ruang parameter yang luas dengan biaya komputasi yang terkontrol, namun tidak menjamin menemukan solusi optimal global karena sifatnya yang random sampling.

---

## 4.4 Hasil Optimasi dengan Grey Wolf Optimizer (GWO)

Grey Wolf Optimizer (GWO) diimplementasikan sebagai metode utama dalam penelitian ini untuk mencari hyperparameter optimal Decision Tree yang memaksimalkan metrik Recall.

### 4.4.1 Konfigurasi GWO

**Tabel 4.5: Konfigurasi Grey Wolf Optimizer**

| Parameter GWO | Nilai | Keterangan |
|---------------|-------|------------|
| Jumlah Wolves (n) | 10 | Ukuran populasi pencarian |
| Max Iterasi (T) | 20 | Batas maksimum iterasi |
| Dimensi (D) | 2 | max_depth dan min_samples_leaf |
| Lower Bound | [1, 1] | Batas bawah pencarian |
| Upper Bound | [20, 50] | Batas atas pencarian |
| Fitness Function | 1 - Recall | Minimize (1 - Recall) = Maximize Recall |
| Cross-Validation | 5-Fold CV | Validasi setiap kandidat solusi |

### 4.4.2 Proses Konvergensi GWO

Gambar 4.1 (grafik konvergensi) menunjukkan proses pencarian solusi optimal oleh GWO selama 20 iterasi. Pada iterasi awal, best score (1 - Recall) berada di sekitar 0.35 (Recall ~65%), yang menunjukkan bahwa wolves diinisialisasi secara random dalam search space.

**Tabel 4.6: Tracking Konvergensi GWO pada Iterasi Kunci**

| Iterasi | max_depth | min_samples_leaf | Recall | Best Score (1-Recall) |
|---------|-----------|------------------|--------|-----------------------|
| 1 | 12 | 8 | 0.6500 | 0.3500 |
| 5 | 7 | 15 | 0.7200 | 0.2800 |
| 10 | 5 | 19 | 0.7650 | 0.2350 |
| 15 | 4 | 21 | 0.7850 | 0.2150 |
| 20 | 4 | 21 | 0.7850 | 0.2150 |

Dari tabel di atas, terlihat bahwa GWO mencapai konvergensi pada iterasi ke-15 dengan parameter optimal: **max_depth=4** dan **min_samples_leaf=21**. Setelah iterasi ke-15, tidak ada perubahan pada parameter maupun skor, yang mengindikasikan bahwa algoritma telah menemukan solusi optimal lokal yang stabil.

### 4.4.3 Parameter Optimal GWO

**Tabel 4.7: Parameter Optimal dari GWO**

| Parameter | Nilai Optimal |
|-----------|---------------|
| criterion | gini |
| max_depth | 4 |
| min_samples_leaf | 21 |
| **Skor Recall (CV)** | **0.7850** |

### 4.4.4 Pembahasan Optimasi GWO

GWO berhasil menemukan parameter optimal dalam waktu sekitar 2 menit, lebih cepat dibandingkan RandomizedSearchCV (5 menit). Hal ini menunjukkan efisiensi GWO dalam mengeksplorasi search space melalui mekanisme guided search yang terinspirasi dari perilaku berburu serigala.

Improvement yang dicapai GWO pada skor cross-validation Recall adalah:
- **+13.05% dibanding Baseline** (0.6545 → 0.7850)
- **+9.41% dibanding RandomizedSearchCV** (0.6909 → 0.7850)

Parameter yang dihasilkan GWO (max_depth=4, min_samples_leaf=21) menghasilkan model yang sangat sederhana namun efektif. Kedalaman pohon yang hanya 4 level memastikan model tidak overfit, sementara min_samples_leaf=21 mencegah pembuatan leaf node yang terlalu spesifik untuk sampel tertentu.

---

## 4.5 Perbandingan Performa Model pada Data Testing

Setelah ketiga model dilatih dengan parameter masing-masing, evaluasi dilakukan pada data testing yang tidak pernah dilihat sebelumnya untuk mengukur kemampuan generalisasi model.

### 4.5.1 Hasil Evaluasi Metrik

**Tabel 4.8: Perbandingan Metrik Performa pada Data Testing**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline DT | 0.7273 | 0.6102 | 0.6545 | 0.6316 |
| RandomizedSearchCV DT | 0.7338 | 0.6129 | 0.6909 | 0.6496 |
| **GWO-Optimized DT** | 0.7013 | 0.5600 | **0.7636** | 0.6462 |

**Improvement GWO terhadap Baseline:**
- Recall: +10.91% (0.6545 → 0.7636)
- F1-Score: +2.31% (0.6316 → 0.6462)

### 4.5.2 Analisis Confusion Matrix

**Tabel 4.9: Confusion Matrix Ketiga Model pada Data Testing**

**Baseline DT:**
|  | Predicted 0 | Predicted 1 |
|---|-------------|-------------|
| **Actual 0** | 76 (TN) | 23 (FP) |
| **Actual 1** | 19 (FN) | 36 (TP) |

**RandomizedSearchCV DT:**
|  | Predicted 0 | Predicted 1 |
|---|-------------|-------------|
| **Actual 0** | 75 (TN) | 24 (FP) |
| **Actual 1** | 17 (FN) | 38 (TP) |

**GWO-Optimized DT:**
|  | Predicted 0 | Predicted 1 |
|---|-------------|-------------|
| **Actual 0** | 66 (TN) | 33 (FP) |
| **Actual 1** | 13 (FN) | 42 (TP) |

Dari confusion matrix di atas, dapat dilihat bahwa:
- **True Positive (TP)**: GWO berhasil mendeteksi 42 kasus diabetes, tertinggi dibanding baseline (36) dan RandomizedSearch (38)
- **False Negative (FN)**: GWO hanya melewatkan 13 kasus diabetes, terendah dibanding baseline (19) dan RandomizedSearch (17)
- **False Positive (FP)**: GWO menghasilkan 33 false alarm, lebih tinggi dari kedua model lainnya

### 4.5.3 Pembahasan Perbandingan Performa

#### Trade-off Precision vs Recall

Model GWO-Optimized menunjukkan **trade-off** antara Precision dan Recall yang khas dalam masalah klasifikasi:

**Recall Tertinggi (76.36%):**
GWO berhasil mencapai Recall tertinggi, artinya dari 55 pasien yang benar-benar menderita diabetes (TP+FN), model berhasil mendeteksi 42 pasien (76.36%). Ini adalah pencapaian yang sangat penting dalam konteks medis.

**Precision Terendah (56.00%):**
Trade-off dari Recall yang tinggi adalah penurunan Precision. Model GWO memprediksi 75 orang sebagai diabetes (TP+FP), namun hanya 42 yang benar-benar diabetes. Ini menghasilkan Precision 56%, yang berarti 44% prediksi positif adalah false alarm.

#### Justifikasi Trade-off dalam Konteks Medis

Dalam aplikasi screening diabetes, **Recall lebih penting daripada Precision** karena:

1. **False Negative sangat berbahaya**: Pasien yang benar-benar diabetes tetapi diprediksi sehat akan tidak mendapat treatment, yang dapat berakibat fatal.

2. **False Positive dapat ditoleransi**: Pasien yang sehat tetapi diprediksi diabetes akan menjalani tes lanjutan (tes glukosa darah konfirmasi). Ini hanya menambah biaya dan waktu, namun tidak membahayakan nyawa.

3. **Cost-Benefit Analysis**: Biaya untuk menangani False Negative (komplikasi diabetes yang tidak terdeteksi) jauh lebih tinggi daripada biaya False Positive (tes konfirmasi tambahan).

GWO dengan sengaja dioptimasi untuk memaksimalkan Recall, sehingga hasil yang diperoleh sesuai dengan objective function. Penurunan Precision dan Accuracy adalah konsekuensi yang dapat diterima dalam konteks aplikasi medis ini.

#### Perbandingan dengan RandomizedSearchCV

RandomizedSearchCV menghasilkan model yang lebih **balanced** dengan:
- Accuracy tertinggi (73.38%)
- F1-Score tertinggi (64.96%)
- Balance yang baik antara Precision (61.29%) dan Recall (69.09%)

Namun, RandomizedSearchCV **tidak dioptimasi khusus untuk Recall**, sehingga metrik Recall-nya lebih rendah dibanding GWO. Jika objective function diubah menjadi F1-Score atau Accuracy, RandomizedSearchCV mungkin akan lebih unggul.

#### Implikasi Klinis

Model GWO-Optimized dengan Recall 76.36% berarti:
- **Dari 100 pasien diabetes, 76 orang akan terdeteksi** oleh sistem
- **24 orang mungkin lolos dari deteksi** (memerlukan awareness dan screening berkala)
- Sistem ini **cocok sebagai alat screening awal**, bukan diagnosis final

---

## 4.6 Analisis Kompleksitas Model

### 4.6.1 Perbandingan Struktur Pohon Keputusan

**Tabel 4.10: Perbandingan Kompleksitas Model**

| Model | Max Depth Aktual | Jumlah Leaf Nodes | Interpretabilitas |
|-------|------------------|-------------------|-------------------|
| Baseline DT | ~23 | >200 | Sangat Sulit ❌ |
| RandomizedSearchCV DT | 5 | ~30 | Sedang ⚠️ |
| **GWO-Optimized DT** | 4 | ~15 | **Mudah ✅** |

Visualisasi pohon keputusan (Gambar 4.2) menunjukkan perbedaan signifikan dalam kompleksitas struktur:

- **Baseline**: Pohon sangat dalam dan lebar, dengan ratusan node yang membuat interpretasi hampir tidak mungkin.
- **RandomizedSearch**: Pohon lebih teratur dengan kedalaman terbatas, namun masih memiliki banyak cabang.
- **GWO**: Pohon sangat sederhana dengan hanya 4 level, memungkinkan praktisi medis untuk memahami logika keputusan dengan mudah.

### 4.6.2 Implikasi untuk Explainability

Model yang sederhana memiliki keunggulan signifikan dalam konteks aplikasi medis:

1. **Transparency**: Dokter dapat melacak jalur keputusan dari root node hingga leaf node untuk memahami mengapa pasien tertentu diprediksi diabetes.

2. **Validation**: Ahli medis dapat memvalidasi apakah split yang dibuat model sesuai dengan pengetahuan klinis (misalnya, apakah threshold Glucose yang dipilih model masuk akal).

3. **Trust**: Model yang dapat dijelaskan membangun kepercayaan antara sistem dan pengguna, yang krusial untuk adopsi teknologi AI dalam healthcare.

4. **Debugging**: Jika terjadi kesalahan prediksi, pohon sederhana memudahkan identifikasi sumber masalah.

---

## 4.7 Implementasi Sistem Prediksi Interaktif dan Explainable AI

Sistem prediksi diabetes diimplementasikan sebagai aplikasi web interaktif menggunakan framework Streamlit dengan 4 modul utama:

### 4.7.1 Modul Upload & Preprocessing

Modul ini memungkinkan pengguna untuk:
- Upload dataset dalam format CSV
- Melihat statistik data awal
- Menjalankan preprocessing otomatis (handling missing values, imputasi, split, standardisasi)
- Menyimpan preprocessed data untuk digunakan di modul berikutnya

### 4.7.2 Modul Optimasi Model

Modul ini menyediakan 3 tab untuk membandingkan metode optimasi:
- **Tab Baseline**: Training model default Decision Tree
- **Tab RandomizedSearch**: Optimasi dengan RandomizedSearchCV
- **Tab GWO**: Optimasi dengan Grey Wolf Optimizer, dilengkapi visualisasi real-time konvergensi

### 4.7.3 Modul Evaluasi & Perbandingan

Modul ini menampilkan:
- Tabel perbandingan metrik (Accuracy, Precision, Recall, F1-Score)
- Confusion Matrix untuk ketiga model
- Visualisasi pohon keputusan (perbandingan kompleksitas)

### 4.7.4 Modul Prediksi Interaktif dengan Explainable AI (XAI)

Modul ini adalah fitur utama untuk penggunaan praktis sistem:

**Input:**
- Form interaktif untuk memasukkan 8 fitur pasien:
  - Pregnancies, Glucose, BloodPressure, SkinThickness
  - Insulin, BMI, DiabetesPedigreeFunction, Age

**Output:**
1. **Prediksi**: Diabetes atau Tidak Diabetes
2. **Confidence Level**: Persentase keyakinan model (berdasarkan `predict_proba`)
3. **Visualisasi SHAP Values**: Bar plot yang menunjukkan kontribusi setiap fitur terhadap prediksi

#### Explainable AI dengan SHAP Values

SHAP (SHapley Additive exPlanations) digunakan untuk menjelaskan prediksi individual. Untuk setiap prediksi, sistem menampilkan:

- **Bar merah**: Fitur yang mendorong prediksi ke arah "Diabetes"
- **Bar biru**: Fitur yang mendorong prediksi ke arah "Tidak Diabetes"
- **Panjang bar**: Magnitude pengaruh fitur

**Contoh Interpretasi:**
> "Pasien A diprediksi diabetes dengan confidence 85%. Faktor utama yang berkontribusi adalah Glucose (180 mg/dL) yang sangat tinggi dan BMI (35.2) yang masuk kategori obesitas. Meskipun Age (28 tahun) relatif muda dan mendorong ke arah 'tidak diabetes', pengaruh Glucose dan BMI lebih dominan."

#### Manfaat XAI untuk Praktisi Medis

1. **Clinical Validation**: Dokter dapat mengecek apakah fitur yang dominan sesuai dengan guideline medis
2. **Patient Communication**: Visualisasi memudahkan penjelasan kepada pasien tentang faktor risiko mereka
3. **Decision Support**: Membantu dokter memutuskan apakah perlu tes konfirmasi atau intervensi preventif
4. **Educational Tool**: Dapat digunakan untuk melatih tenaga kesehatan memahami faktor risiko diabetes

---

## 4.8 Kelebihan dan Keterbatasan Penelitian

### 4.8.1 Kelebihan

1. **Optimasi Berbasis Metaheuristik**: Penggunaan GWO memberikan alternatif yang efisien dibanding metode tradisional seperti Grid Search atau Random Search.

2. **Fokus pada Metrik Klinis**: Pemilihan Recall sebagai objective function sesuai dengan prioritas dalam aplikasi medical screening.

3. **Model Interpretable**: Pohon keputusan yang sederhana (depth=4) memungkinkan explainability yang penting dalam healthcare.

4. **Sistem End-to-End**: Implementasi aplikasi web interaktif memungkinkan penggunaan langsung oleh praktisi.

5. **Explainable AI**: Integrasi SHAP values memberikan transparansi pada level prediksi individual.

### 4.8.2 Keterbatasan

1. **Dataset Terbatas**: Hanya menggunakan Pima Indians Diabetes Database (768 sampel) yang mungkin tidak representatif untuk populasi global.

2. **Single Algorithm**: Hanya menggunakan Decision Tree sebagai classifier. Algoritma lain seperti Random Forest, XGBoost, atau Neural Networks mungkin memberikan performa lebih baik.

3. **Hyperparameter Terbatas**: Hanya mengoptimasi 2 hyperparameter (max_depth, min_samples_leaf). Parameter lain seperti min_samples_split, max_features, dll tidak dioptimasi.

4. **Evaluasi Single Split**: Menggunakan single train-test split (80:20) alih-alih nested cross-validation yang lebih robust.

5. **Tidak Ada External Validation**: Model tidak divalidasi pada dataset eksternal dari sumber berbeda.

6. **Trade-off Precision**: Precision yang rendah (56%) mungkin menyebabkan beban berlebih pada sistem kesehatan akibat banyaknya false positive yang perlu tes lanjutan.

---

## 4.9 Ringkasan Hasil

Penelitian ini berhasil mengimplementasikan sistem prediksi diabetes menggunakan Decision Tree yang dioptimasi dengan Grey Wolf Optimizer. Hasil utama yang dicapai:

1. **GWO menemukan parameter optimal** (max_depth=4, min_samples_leaf=21) yang menghasilkan Recall 76.36%, meningkat 10.91% dibanding baseline.

2. **GWO lebih efisien** dibanding RandomizedSearchCV dalam hal waktu komputasi (2 menit vs 5 menit) sambil mencapai Recall lebih tinggi.

3. **Trade-off Precision-Recall** yang terjadi pada model GWO dapat dibenarkan dalam konteks medical screening dimana Recall adalah prioritas utama.

4. **Model sederhana dan interpretable** memungkinkan validasi oleh ahli medis dan membangun trust dalam penggunaan klinis.

5. **Implementasi XAI dengan SHAP** memberikan transparansi pada prediksi individual, mendukung clinical decision making.

Hasil penelitian ini menunjukkan bahwa optimasi metaheuristik dengan GWO merupakan pendekatan yang viable untuk hyperparameter tuning dalam konteks aplikasi medis yang memprioritaskan sensitivitas deteksi.
