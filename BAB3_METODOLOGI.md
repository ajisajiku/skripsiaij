
# BAB 3: METODOLOGI PENELITIAN

## 3.1 Pendahuluan

Metodologi penelitian ini menguraikan langkah-langkah sistematis yang dilakukan untuk membangun model prediksi penyakit diabetes menggunakan pendekatan hybrid Grey Wolf Optimizer (GWO) dan Decision Tree. Penelitian ini mengadopsi metode kuantitatif dengan pendekatan eksperimental untuk mengembangkan sistem prediksi yang akurat dan efisien.

## 3.2 Jenis Penelitian

Penelitian ini termasuk dalam kategori **penelitian terapan (applied research)** dengan pendekatan **kuantitatif**. Penelitian terapan dipilih karena bertujuan untuk menghasilkan solusi praktis dalam bentuk model prediksi yang dapat diimplementasikan dalam dunia medis. Pendekatan kuantitatif digunakan karena data yang dianalisis bersifat numerik dan memerlukan analisis statistik serta komputasi.

## 3.3 Waktu dan Tempat Penelitian

- **Waktu Pelaksanaan:** [Tahun] - [Tahun]
- **Tempat Penelitian:** Laboratorium Komputasi dan Analisis Data
- **Sumber Data:** Dataset publik dari Kaggle (Pima Indians Diabetes Database)

## 3.4 Alur Penelitian

Tahapan penelitian dilakukan secara sistematis mengikuti diagram alir berikut:

1. **Identifikasi Masalah** → 2. **Studi Literatur** → 3. **Pengumpulan Data** → 4. **Preprocessing Data** → 5. **Implementasi Algoritma** → 6. **Eksperimen** → 7. **Analisis Hasil** → 8. **Kesimpulan**

## 3.5 Data dan Sumber Data

### 3.5.1 Dataset

Penelitian ini menggunakan dataset **Pima Indians Diabetes Database** yang diperoleh dari platform Kaggle. Dataset ini merupakan data rekam medis yang umum digunakan untuk masalah klasifikasi biner, yaitu memprediksi apakah seorang pasien menderita diabetes atau tidak berdasarkan pengukuran diagnostik tertentu.

Dataset ini terdiri dari beberapa atribut prediktor (fitur) dan satu atribut target (kelas), yaitu:

**Tabel 3.1: Atribut Dataset Pima Indians Diabetes**

| No | Nama Atribut | Tipe Data | Deskripsi |
|----|--------------|-----------|-----------|
| 1 | Pregnancies | Numerik | Jumlah kali hamil |
| 2 | Glucose | Numerik | Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral |
| 3 | BloodPressure | Numerik | Tekanan darah diastolik (mm Hg) |
| 4 | SkinThickness | Numerik | Ketebalan lipatan kulit trisep (mm) |
| 5 | Insulin | Numerik | Insulin serum 2 jam (μU/ml) |
| 6 | BMI | Numerik | Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)²) |
| 7 | DiabetesPedigreeFunction | Numerik | Fungsi pedigree diabetes |
| 8 | Age | Numerik | Usia (tahun) |
| 9 | Outcome | Kategorik | Variabel target (0 = tidak diabetes, 1 = diabetes) |

**Karakteristik Dataset:**
- **Jumlah Sampel:** 768 pasien
- **Jumlah Fitur:** 8 atribut prediktor
- **Variabel Target:** 1 atribut biner
- **Kelas Seimbang:** 500 non-diabetes (65.1%), 268 diabetes (34.9%)

## 3.6 Preprocessing Data

Tahap preprocessing bertujuan untuk membersihkan dan mempersiapkan data agar sesuai untuk proses pemodelan. Preprocessing merupakan tahap krusial dalam penelitian data mining untuk memastikan kualitas data yang optimal.

### 3.6.1 Identifikasi Masalah Data

Sebelum melakukan preprocessing, dilakukan identifikasi masalah pada dataset:
1. **Missing Values:** Nilai nol (0) pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI`
2. **Outliers:** Nilai ekstrem pada beberapa atribut
3. **Skala yang Berbeda:** Variabel dengan rentang nilai yang bervariasi

### 3.6.2 Teknik Preprocessing

Langkah-langkah preprocessing yang dilakukan:

1.  **Pembersihan Data (Data Cleaning):**
    - Nilai nol (0) pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` dianggap sebagai missing values
    - Nilai-nilai ini diganti dengan `NaN` (Not a Number) untuk proses imputasi selanjutnya

2.  **Imputasi Data (Data Imputation):**
    - Strategi imputasi menggunakan **median** dari setiap kolom
    - Median dipilih karena robust terhadap outliers
    - Rumus: `X_imputed = median(X_non_missing)`

3.  **Pemisahan Data (Data Splitting):**
    - Dataset dibagi menggunakan stratified sampling
    - 80% data untuk pelatihan (*training set*): 614 sampel
    - 20% data untuk pengujian (*testing set*): 154 sampel
    - Stratifikasi mempertahankan proporsi kelas pada kedua set

4.  **Standardisasi Fitur (Feature Scaling):**
    - Menggunakan `StandardScaler` dari library scikit-learn
    - Transformasi: `z = (x - μ) / σ`
    - Dimana μ adalah mean dan σ adalah standar deviasi
    - Tujuan: membuat semua fitur memiliki skala yang sama (mean=0, std=1)

## 3.7 Metode dan Algoritma

### 3.7.1 Grey Wolf Optimizer (GWO)

Grey Wolf Optimizer (GWO) adalah algoritma optimisasi metaheuristik yang dikembangkan oleh Mirjalili et al. (2014). Algoritma ini terinspirasi dari hierarki sosial dan perilaku berburu serigala abu-abu di alam liar.

#### 3.7.1.1 Hierarki Sosial Serigala

Dalam struktur sosial serigala abu-abu, terdapat empat tingkatan hierarki:

1. **Alpha (α):** Pemimpin kelompok, individu terbaik dalam populasi
2. **Beta (β):** Pembantu alpha, solusi kedua terbaik
3. **Delta (δ):** Subordinat, solusi ketiga terbaik
4. **Omega (ω):** Serigala berperingkat terendah, sisa populasi

#### 3.7.1.2 Tahapan Berburu GWO

Algoritma GWO mensimulasikan tiga tahapan utama berburu:

1. **Menguntit (Tracking):** Serigala mengidentifikasi dan mengikuti mangsa
2. **Mengepung (Encircling):** Alpha, beta, dan delta memimpin pengepungan mangsa
3. **Menyerang (Attacking):** Serigala omega mengeksploitasi solusi terbaik

#### 3.7.1.3 Implementasi Matematis

**Pengepungan Mangsa:**
```
D = |C ⋅ X_p(t) - X(t)|
X(t+1) = X_p(t) - A ⋅ D
```

Dimana:
- `X_p`: Posisi mangsa (solusi terbaik)
- `X`: Posisi serigala (solusi kandidat)
- `A` dan `C`: Koefisien vektor

**Koefisien Vektor:**
```
A = 2a ⋅ r1 - a
C = 2 ⋅ r2
```

Dimana:
- `a`: Parameter yang menurun secara linier dari 2 ke 0
- `r1`, `r2`: Vektor acak dalam [0,1]

#### 3.7.1.4 Parameter GWO

Parameter GWO yang digunakan dalam penelitian:
- **Ukuran Populasi:** 30 serigala
- **Iterasi Maksimum:** 50 generasi
- **Dimensi Pencarian:** 2 (max_depth, min_samples_leaf)
- **Fungsi Fitness:** 1 - Recall (untuk minimisasi)

### 3.7.2 Decision Tree Classifier

Decision Tree adalah algoritma pembelajaran terawasi yang membangun model klasifikasi dalam bentuk struktur pohon. Setiap node internal mewakili tes pada atribut, setiap cabang mewakili hasil tes, dan setiap node daun mewakili label kelas.

#### 3.7.2.1 Konsep Dasar

**Prinsip Pembelajaran:**
- Menggunakan algoritma rekursif untuk membagi data
- Memilih atribut terbaik untuk setiap pembagian
- Memaksimalkan informasi gain atau minimisasi impurity

**Kriteria Pembagian:**
- **Gini Index:** Mengukur ketidakmurnian sampel dalam node
- **Information Gain:** Berdasarkan entropy dari informasi
- **Chi-Square:** Uji statistik untuk independensi

#### 3.7.2.2 Hyperparameter yang Dioptimalkan

1. **max_depth:** Kedalaman maksimum pohon
   - Rentang: [1, 20]
   - Mencegah overfitting

2. **min_samples_leaf:** Jumlah minimum sampel di node daun
   - Rentang: [1, 50]
   - Mengontrol kompleksitas model

3. **criterion:** Kriteria pembagian
   - 'gini' atau 'entropy'
   - Ditetapkan sebagai 'gini' untuk konsistensi

### 3.7.3 Arsitektur Hybrid GWO-Decision Tree

Penelitian ini menggabungkan GWO dan Decision Tree dengan arsitektur berikut:

1. **GWO sebagai Optimisator Hyperparameter**
   - Mencari kombinasi optimal max_depth dan min_samples_leaf
   - Menggunakan validasi silang 5-fold untuk evaluasi

2. **Decision Tree sebagai Klasifikasi**
   - Menggunakan hyperparameter optimal dari GWO
   - Membangun model klasifikasi final

## 3.8 Implementasi Sistem

### 3.8.1 Lingkungan Pengembangan

**Sistem Operasi:**
- Windows 10/11 atau Linux Ubuntu 20.04+

**Python dan Library:**
- Python 3.8+
- scikit-learn 1.0.2
- numpy 1.21.6
- pandas 1.4.2
- matplotlib 3.5.2
- seaborn 0.11.2

**Hardware Rekomendasi:**
- Processor: Intel Core i5 atau AMD Ryzen 5 ke atas
- RAM: Minimum 8GB
- Storage: 2GB ruang kosong

### 3.8.2 Arsitektur Sistem

**Struktur Modul:**
1. **Data Loader Module:** Memuat dan validasi dataset
2. **Preprocessing Module:** Bersihkan dan transformasi data
3. **GWO Optimizer Module:** Optimisasi hyperparameter
4. **Decision Tree Module:** Klasifikasi dan prediksi
5. **Evaluation Module:** Evaluasi performa model
6. **Visualization Module:** Visualisasi hasil

### 3.8.3 Alur Kerja Implementasi

1. **Inisialisasi Parameter:**
   - Set parameter GWO (population size, max iterations)
   - Define rentang hyperparameter Decision Tree

2. **Optimisasi GWO:**
   - Inisialisasi populasi serigala acak
   - Evaluasi fitness untuk setiap individu
   - Update posisi alpha, beta, delta
   - Iterasi hingga konvergensi

3. **Pelatihan Model Final:**
   - Gunakan hyperparameter optimal
   - Latih Decision Tree pada training set
   - Validasi pada testing set

## 3.9 Evaluasi Model

### 3.9.1 Metrik Evaluasi

**1. Accuracy (Akurasi):**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Mengukur proporsi prediksi yang benar terhadap total prediksi.

**2. Precision:**
```
Precision = TP / (TP + FP)
```
Mengukur propeksi positif yang benar dari semua prediksi positif.

**3. Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
Mengukur kemampuan mendeteksi kasus positif (metrik utama dalam penelitian ini).

**4. F1-Score:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```
Rata-rata harmonis antara precision dan recall.

**5. Specificity:**
```
Specificity = TN / (TN + FP)
```
Mengukur kemampuan mendeteksi kasus negatif.

### 3.9.2 Confusion Matrix

Matriks konfusi digunakan untuk visualisasi performa klasifikasi:

| Prediksi \ Aktual | Diabetes (1) | Tidak Diabetes (0) |
|-------------------|--------------|-------------------|
| Diabetes (1)      | True Positive (TP) | False Positive (FP) |
| Tidak Diabetes (0) | False Negative (FN) | True Negative (TN) |

### 3.9.3 Validasi Silang (Cross-Validation)

Menggunakan **5-Fold Cross-Validation** untuk evaluasi yang robust:

1. Dataset dibagi menjadi 5 subset yang sama besar
2. Model dilatih pada 4 subset dan diuji pada 1 subset
3. Proses diulang 5 kali dengan subset uji yang berbeda
4. Hasil rata-rata dari 5 eksperimen digunakan sebagai evaluasi final

**Keuntungan 5-Fold CV:**
- Mengurangi varians evaluasi
- Memaksimalkan penggunaan data
- Menghindari overfitting pada data split tertentu

### 3.9.4 Kriteria Evaluasi

**Prioritas Metrik:**
1. **Recall:** Prioritas utama (minimalkan False Negative)
2. **F1-Score:** Keseimbangan precision dan recall
3. **Accuracy:** Akurasi keseluruhan
4. **AUC-ROC:** Performa klasifikasi biner

**Alasan Prioritas Recall:**
- Dalam diagnosis diabetes, False Negative (pasien diabetes yang tidak terdeteksi) lebih berbahaya
- Biaya pengobatan dini lebih rendah daripada komplikasi diabetes lanjut

## 3.10 Analisis Komparatif

### 3.10.1 Model Baseline

Untuk mengevaluasi keunggulan GWO-Decision Tree, dilakukan perbandingan dengan:

1. **Decision Tree Standar:** Tanpa optimisasi hyperparameter
2. **Random Forest:** Ensemble method
3. **Support Vector Machine (SVM):** Classical machine learning
4. **Logistic Regression:** Baseline model sederhana
5. **Grid Search-Decision Tree:** Optimisasi hyperparameter konvensional

### 3.10.2 Uji Hipotesis

**Hipotesis Nol (H0):** Tidak ada perbedaan signifikan antara GWO-Decision Tree dengan model baseline

**Hipotesis Alternatif (H1):** GWO-Decision Tree memiliki performa signifikan lebih baik

**Uji Statistik:** Paired t-test dengan α = 0.05

## 3.11 Etika Penelitian

### 3.11.1 Pertimbangan Etika

1. **Privasi Data:** Dataset yang digunakan bersifat publik dan anonim
2. **Validitas Hasil:** Implementasi mengikuti best practices machine learning
3. **Reproduktibilitas:** Kode dan data tersedia untuk verifikasi
4. **Transparansi:** Metodologi dipublikasikan secara lengkap

### 3.11.2 Keterbatasan Penelitian

1. **Dataset:** Terbatas pada populasi Pima Indians
2. **Fitur:** Hanya 8 fitur medis dasar
3. **Generalisasi:** Perlu validasi pada dataset yang lebih beragam
4. **Interpretasi:** Model perlu interpretasi medis lebih lanjut
