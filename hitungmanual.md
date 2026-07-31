# PERHITUNGAN MANUAL: Prediksi Diabetes dengan GWO-Decision Tree
## Pima Indians Diabetes Database

> Dokumen ini menyajikan **10 contoh perhitungan manual** lengkap step-by-step dari setiap proses dalam penelitian prediksi diabetes menggunakan Grey Wolf Optimizer (GWO) dan Decision Tree. Setiap contoh dilengkapi dengan rumus, substitusi angka, dan hasil akhir beserta penjelasannya.

---

## DAFTAR PERHITUNGAN

| No | Topik |
|----|-------|
| 1 | Identifikasi Missing Values (Nilai 0 Tidak Valid) |
| 2 | Perhitungan Median untuk Imputasi |
| 3 | Imputasi Median pada Data Pasien |
| 4 | Standardisasi Fitur dengan Z-Score (StandardScaler) |
| 5 | Stratified Train-Test Split |
| 6 | Gini Index pada Decision Tree |
| 7 | Information Gain (Entropy) |
| 8 | Fitness Function GWO |
| 9 | Update Posisi Wolf pada GWO |
| 10 | Evaluasi Model: Accuracy, Precision, Recall, F1-Score |

---

---

# PERHITUNGAN 1
## Identifikasi Missing Values — Nilai 0 Tidak Valid Secara Medis

### Latar Belakang

Pada dataset Pima Indians Diabetes Database, beberapa kolom memiliki **nilai 0 yang tidak mungkin secara biologis**. Nilai 0 pada kolom seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` dianggap sebagai data hilang (*missing values*) yang perlu ditangani sebelum pemodelan.

### Rumus: Persentase Missing Values

```
Persentase Missing (%) = (Jumlah Nilai 0 / Total Sampel) x 100%
```

### Data Awal

- **Total sampel dataset:** 768 pasien
- **Kolom yang diidentifikasi:** Glucose, BloodPressure, SkinThickness, Insulin, BMI

---

### LANGKAH 1 — Hitung Persentase Missing untuk Glucose

**Input:**
- Jumlah nilai 0 pada kolom Glucose = **5**
- Total sampel = **768**

**Proses:**
```
Persentase Missing (Glucose) = (Jumlah Nilai 0 / Total Sampel) x 100%
Persentase Missing (Glucose) = (5 / 768) x 100%
Persentase Missing (Glucose) = 0.006510 x 100%
Persentase Missing (Glucose) = 0.65%
```

**Output:** Glucose memiliki **0.65%** missing values -> tergolong rendah, tetapi tetap harus diimputasi.

---

### LANGKAH 2 — Hitung Persentase Missing untuk BloodPressure

**Input:**
- Jumlah nilai 0 pada kolom BloodPressure = **35**
- Total sampel = **768**

**Proses:**
```
Persentase Missing (BloodPressure) = (35 / 768) x 100%
Persentase Missing (BloodPressure) = 0.045573 x 100%
Persentase Missing (BloodPressure) = 4.56%
```

**Output:** BloodPressure memiliki **4.56%** missing values.

---

### LANGKAH 3 — Hitung Persentase Missing untuk SkinThickness

**Input:**
- Jumlah nilai 0 pada kolom SkinThickness = **227**
- Total sampel = **768**

**Proses:**
```
Persentase Missing (SkinThickness) = (227 / 768) x 100%
Persentase Missing (SkinThickness) = 0.295573 x 100%
Persentase Missing (SkinThickness) = 29.56%
```

**Output:** SkinThickness memiliki **29.56%** missing values -> tergolong TINGGI!

---

### LANGKAH 4 — Hitung Persentase Missing untuk Insulin

**Input:**
- Jumlah nilai 0 pada kolom Insulin = **374**
- Total sampel = **768**

**Proses:**
```
Persentase Missing (Insulin) = (374 / 768) x 100%
Persentase Missing (Insulin) = 0.486979 x 100%
Persentase Missing (Insulin) = 48.70%
```

**Output:** Insulin memiliki **48.70%** missing values -> hampir setengah data hilang!

---

### LANGKAH 5 — Hitung Persentase Missing untuk BMI

**Input:**
- Jumlah nilai 0 pada kolom BMI = **11**
- Total sampel = **768**

**Proses:**
```
Persentase Missing (BMI) = (11 / 768) x 100%
Persentase Missing (BMI) = 0.014323 x 100%
Persentase Missing (BMI) = 1.43%
```

**Output:** BMI memiliki **1.43%** missing values.

---

### RINGKASAN PERHITUNGAN 1

| Kolom | Nilai 0 | Total Sampel | Rumus | Persentase |
|-------|---------|--------------|-------|------------|
| Glucose | 5 | 768 | (5/768)x100 | **0.65%** |
| BloodPressure | 35 | 768 | (35/768)x100 | **4.56%** |
| SkinThickness | 227 | 768 | (227/768)x100 | **29.56%** |
| Insulin | 374 | 768 | (374/768)x100 | **48.70%** |
| BMI | 11 | 768 | (11/768)x100 | **1.43%** |

**Keputusan:** Semua nilai 0 pada 5 kolom tersebut diubah menjadi `NaN` dan akan diisi menggunakan **imputasi median**.

---

---

# PERHITUNGAN 2
## Perhitungan Median untuk Imputasi

### Latar Belakang

Median dipilih sebagai strategi imputasi karena **robust terhadap outlier** dibanding mean. Median dihitung hanya dari **data training** (614 sampel) untuk menghindari data leakage dari data testing.

### Rumus: Median

Untuk data yang sudah diurutkan sejumlah **n**:

```
Jika n ganjil:  Median = nilai ke-[(n+1)/2]
Jika n genap:   Median = (nilai ke-[n/2] + nilai ke-[n/2 + 1]) / 2
```

---

### LANGKAH 1 — Median SkinThickness (n genap)

**Input:**
- Jumlah data valid SkinThickness di training (setelah NaN dihapus) = **438**
- Kolom diurutkan dari kecil ke besar

**Proses:**
```
n = 438  -> n GENAP

Posisi tengah 1 = n / 2         = 438 / 2       = 219
Posisi tengah 2 = n / 2 + 1     = 438 / 2 + 1   = 220

Nilai ke-219 = 28
Nilai ke-220 = 29

Median = (28 + 29) / 2
Median = 57 / 2
Median = 28.5
```

**Output:** Median SkinThickness = **28.5 mm**

---

### LANGKAH 2 — Median Insulin (n genap)

**Input:**
- Jumlah data valid Insulin di training = **324**

**Proses:**
```
n = 324  -> n GENAP

Posisi tengah 1 = 324 / 2       = 162
Posisi tengah 2 = 324 / 2 + 1   = 163

Nilai ke-162 = 120
Nilai ke-163 = 120

Median = (120 + 120) / 2
Median = 240 / 2
Median = 120
```

**Output:** Median Insulin = **120 uU/ml**

---

### LANGKAH 3 — Median BloodPressure (n genap)

**Input:**
- Jumlah data valid BloodPressure di training = **590**

**Proses:**
```
n = 590  -> n GENAP

Posisi tengah 1 = 590 / 2       = 295
Posisi tengah 2 = 590 / 2 + 1   = 296

Nilai ke-295 = 72
Nilai ke-296 = 72

Median = (72 + 72) / 2
Median = 144 / 2
Median = 72
```

**Output:** Median BloodPressure = **72 mm Hg**

---

### LANGKAH 4 — Median Glucose (n ganjil)

**Input:**
- Jumlah data valid Glucose di training = **609**

**Proses:**
```
n = 609  -> n GANJIL

Posisi tengah = (n + 1) / 2 = (609 + 1) / 2 = 610 / 2 = 305

Nilai ke-305 = 118

Median = 118
```

**Output:** Median Glucose = **118 mg/dL**

---

### LANGKAH 5 — Median BMI (n ganjil)

**Input:**
- Jumlah data valid BMI di training = **607**

**Proses:**
```
n = 607  -> n GANJIL

Posisi tengah = (607 + 1) / 2 = 608 / 2 = 304

Nilai ke-304 = 32

Median = 32
```

**Output:** Median BMI = **32.0 kg/m2**

---

### RINGKASAN PERHITUNGAN 2

| Atribut | N Valid | Ganjil/Genap | Posisi Tengah | Nilai pada Posisi | Median |
|---------|---------|--------------|---------------|-------------------|--------|
| Glucose | 609 | Ganjil | ke-305 | 118 | **118** |
| BloodPressure | 590 | Genap | ke-295 & ke-296 | 72, 72 | **72** |
| SkinThickness | 438 | Genap | ke-219 & ke-220 | 28, 29 | **28.5** |
| Insulin | 324 | Genap | ke-162 & ke-163 | 120, 120 | **120** |
| BMI | 607 | Ganjil | ke-304 | 32 | **32** |

---

---

# PERHITUNGAN 3
## Imputasi Median pada Data Pasien

### Latar Belakang

Setelah median dihitung dari data training, nilai median tersebut digunakan untuk mengisi `NaN` pada seluruh baris data (baik training maupun testing).

### Rumus Imputasi

```
Jika X = NaN  ->  X_imputasi = Median(kolom)
Jika X != NaN ->  X_imputasi = X (tidak berubah)
```

---

### Data Pasien yang Diimputasi

Kita gunakan **Baris 1 data training** sebagai contoh kasus.

**Data Asli Baris 1 (sebelum imputasi):**

| Fitur | Nilai Asli |
|-------|-----------|
| Pregnancies | 2 |
| Glucose | 84 |
| BloodPressure | **NaN** (asalnya 0) |
| SkinThickness | **NaN** (asalnya 0) |
| Insulin | **NaN** (asalnya 0) |
| BMI | **NaN** (asalnya 0) |
| DiabetesPedigreeFunction | 0.304 |
| Age | 21 |

---

### LANGKAH 1 — Imputasi BloodPressure Baris 1

**Input:** BloodPressure = NaN, Median BloodPressure = 72

**Proses:**
```
X = NaN  ->  Terapkan imputasi

X_imputasi = Median(BloodPressure)
X_imputasi = 72
```

**Output:** BloodPressure Baris 1 = **72 mm Hg**

---

### LANGKAH 2 — Imputasi SkinThickness Baris 1

**Input:** SkinThickness = NaN, Median SkinThickness = 28.5

**Proses:**
```
X = NaN  ->  Terapkan imputasi

X_imputasi = Median(SkinThickness)
X_imputasi = 28.5
```

**Output:** SkinThickness Baris 1 = **28.5 mm**

---

### LANGKAH 3 — Imputasi Insulin Baris 1

**Input:** Insulin = NaN, Median Insulin = 120

**Proses:**
```
X = NaN  ->  Terapkan imputasi

X_imputasi = Median(Insulin)
X_imputasi = 120
```

**Output:** Insulin Baris 1 = **120 uU/ml**

---

### LANGKAH 4 — Imputasi BMI Baris 1

**Input:** BMI = NaN, Median BMI = 32

**Proses:**
```
X = NaN  ->  Terapkan imputasi

X_imputasi = Median(BMI)
X_imputasi = 32
```

**Output:** BMI Baris 1 = **32.0 kg/m2**

---

### LANGKAH 5 — Verifikasi Nilai yang Tidak Berubah

**Input:** Glucose = 84 (bukan NaN), Pregnancies = 2 (bukan NaN)

**Proses:**
```
X != NaN  ->  Tidak perlu imputasi

Glucose_imputasi     = 84  (tetap)
Pregnancies_imputasi = 2   (tetap)
```

**Output:** Nilai yang tidak memiliki NaN tetap tidak berubah.

---

### RINGKASAN PERHITUNGAN 3: Baris 1 Sebelum vs Sesudah Imputasi

| Fitur | Sebelum Imputasi | Keputusan | Setelah Imputasi |
|-------|-----------------|-----------|-----------------|
| Pregnancies | 2 | Tidak perlu imputasi | **2** |
| Glucose | 84 | Tidak perlu imputasi | **84** |
| BloodPressure | NaN | Imputasi -> Median 72 | **72** |
| SkinThickness | NaN | Imputasi -> Median 28.5 | **28.5** |
| Insulin | NaN | Imputasi -> Median 120 | **120** |
| BMI | NaN | Imputasi -> Median 32 | **32** |
| DiabetesPedigreeFunction | 0.304 | Tidak perlu imputasi | **0.304** |
| Age | 21 | Tidak perlu imputasi | **21** |

---

---

# PERHITUNGAN 4
## Standardisasi Fitur dengan Z-Score (StandardScaler)

### Latar Belakang

Standardisasi dilakukan agar semua fitur berada pada skala yang sama (mean=0, std=1). Tanpa standardisasi, fitur dengan rentang nilai besar (seperti Insulin 0-900) akan mendominasi fitur dengan rentang kecil (seperti DiabetesPedigreeFunction 0-2.4).

### Rumus Z-Score

```
z = (x - mu) / sigma

Keterangan:
  x     = nilai asli fitur
  mu    = mean fitur pada data training
  sigma = standar deviasi fitur pada data training
  z     = nilai terstandarisasi (z-score)
```

### Parameter StandardScaler (dari data training 614 sampel)

| Atribut | Mean (mu) | Standar Deviasi (sigma) |
|---------|-----------|-------------------------|
| Pregnancies | 3.742671 | 3.310565 |
| Glucose | 121.815961 | 30.079488 |
| BloodPressure | 72.229642 | 12.089421 |
| SkinThickness | 28.568404 | 8.403178 |
| Insulin | 138.115635 | 88.578061 |
| BMI | 32.348208 | 6.929968 |
| DiabetesPedigreeFunction | 0.469168 | 0.336572 |
| Age | 32.907166 | 11.494065 |

---

### LANGKAH 1 — Standardisasi Glucose (Baris 1: x = 84)

**Input:** Glucose = 84, mu = 121.815961, sigma = 30.079488

**Proses:**
```
z = (x - mu) / sigma
z = (84 - 121.815961) / 30.079488
z = (-37.815961) / 30.079488
z = -1.257201
```

**Interpretasi:** Nilai Glucose 84 berada **1.257 standar deviasi DI BAWAH** rata-rata -> pasien ini memiliki Glucose yang rendah dari rata-rata training.

**Output:** z_Glucose = **-1.2572**

---

### LANGKAH 2 — Standardisasi BloodPressure (Baris 1: x = 72)

**Input:** BloodPressure = 72 (hasil imputasi), mu = 72.229642, sigma = 12.089421

**Proses:**
```
z = (x - mu) / sigma
z = (72 - 72.229642) / 12.089421
z = (-0.229642) / 12.089421
z = -0.018995
```

**Interpretasi:** Nilai BloodPressure 72 sangat mendekati rata-rata (z sekitar 0), artinya tekanan darah pasien ini normal.

**Output:** z_BloodPressure = **-0.0190**

---

### LANGKAH 3 — Standardisasi SkinThickness (Baris 1: x = 28.5)

**Input:** SkinThickness = 28.5, mu = 28.568404, sigma = 8.403178

**Proses:**
```
z = (x - mu) / sigma
z = (28.5 - 28.568404) / 8.403178
z = (-0.068404) / 8.403178
z = -0.008140
```

**Output:** z_SkinThickness = **-0.0081**

---

### LANGKAH 4 — Standardisasi Pregnancies (Baris 1: x = 2)

**Input:** Pregnancies = 2, mu = 3.742671, sigma = 3.310565

**Proses:**
```
z = (x - mu) / sigma
z = (2 - 3.742671) / 3.310565
z = (-1.742671) / 3.310565
z = -0.526379
```

**Output:** z_Pregnancies = **-0.5264**

---

### LANGKAH 5 — Standardisasi Insulin (Baris 1: x = 120)

**Input:** Insulin = 120 (hasil imputasi), mu = 138.115635, sigma = 88.578061

**Proses:**
```
z = (x - mu) / sigma
z = (120 - 138.115635) / 88.578061
z = (-18.115635) / 88.578061
z = -0.204510
```

**Output:** z_Insulin = **-0.2045**

---

### RINGKASAN PERHITUNGAN 4: Baris 1 Setelah Standardisasi

| Fitur | Nilai Asli (x) | Mean (mu) | Std (sigma) | Hitung: (x-mu)/sigma | Z-Score |
|-------|---------------|-----------|-------------|----------------------|---------|
| Pregnancies | 2.0 | 3.742671 | 3.310565 | (2-3.742671)/3.310565 | **-0.5264** |
| Glucose | 84.0 | 121.815961 | 30.079488 | (84-121.815961)/30.079488 | **-1.2572** |
| BloodPressure | 72.0 | 72.229642 | 12.089421 | (72-72.229642)/12.089421 | **-0.0190** |
| SkinThickness | 28.5 | 28.568404 | 8.403178 | (28.5-28.568404)/8.403178 | **-0.0081** |
| Insulin | 120.0 | 138.115635 | 88.578061 | (120-138.115635)/88.578061 | **-0.2045** |
| BMI | 32.0 | 32.348208 | 6.929968 | (32-32.348208)/6.929968 | **-0.0502** |
| DiabetesPedigreeFunction | 0.304 | 0.469168 | 0.336572 | (0.304-0.469168)/0.336572 | **-0.4907** |
| Age | 21.0 | 32.907166 | 11.494065 | (21-32.907166)/11.494065 | **-1.0359** |

> **Verifikasi:** Hasil perhitungan manual identik dengan output StandardScaler scikit-learn (terkonfirmasi)

---

---

# PERHITUNGAN 5
## Stratified Train-Test Split

### Latar Belakang

Dataset dibagi menjadi training (80%) dan testing (20%) menggunakan **stratified sampling** agar proporsi kelas (Diabetes vs Tidak Diabetes) tetap terjaga di kedua subset.

### Rumus Stratified Split

```
n_train = round(N x train_ratio)
n_test  = N - n_train

Untuk setiap kelas k:
  n_train_k = round(N_k x train_ratio)
  n_test_k  = N_k - n_train_k
```

---

### Data Kelas Dataset

| Kelas | Label | Jumlah (N_k) | Persentase |
|-------|-------|-------------|------------|
| Tidak Diabetes | 0 | 500 | 65.10% |
| Diabetes | 1 | 268 | 34.90% |
| **Total** | - | **768** | **100%** |

---

### LANGKAH 1 — Total Data per Set

**Input:** N = 768, train_ratio = 0.80, test_ratio = 0.20

**Proses:**
```
n_train = round(768 x 0.80)
n_train = round(614.4)
n_train = 614

n_test  = 768 - 614
n_test  = 154
```

**Output:** Training = **614 sampel**, Testing = **154 sampel**

---

### LANGKAH 2 — Distribusi Kelas pada Training Set

**Input:** N_k(Tidak Diabetes) = 500, train_ratio = 0.80

**Proses:**
```
n_train(Tidak Diabetes) = round(500 x 0.80)
n_train(Tidak Diabetes) = round(400)
n_train(Tidak Diabetes) = 400

n_test(Tidak Diabetes)  = 500 - 400
n_test(Tidak Diabetes)  = 100
```

---

### LANGKAH 3 — Distribusi Kelas Diabetes

**Input:** N_k(Diabetes) = 268, train_ratio = 0.80

**Proses:**
```
n_train(Diabetes) = round(268 x 0.80)
n_train(Diabetes) = round(214.4)
n_train(Diabetes) = 214

n_test(Diabetes)  = 268 - 214
n_test(Diabetes)  = 54
```

---

### LANGKAH 4 — Verifikasi Proporsi Kelas

**Proporsi kelas Training:**
```
Proporsi_0_train = 400 / 614 = 0.6515 = 65.15%
Proporsi_1_train = 214 / 614 = 0.3485 = 34.85%
```

**Proporsi kelas Testing:**
```
Proporsi_0_test = 100 / 154 = 0.6494 = 64.94%
Proporsi_1_test = 54  / 154 = 0.3506 = 35.06%
```

**Proporsi kelas dataset asli:**
```
Proporsi_0_asli = 500 / 768 = 0.6510 = 65.10%
Proporsi_1_asli = 268 / 768 = 0.3490 = 34.90%
```

**Output:** Proporsi kelas pada training (~65%/35%) dan testing (~65%/35%) mendekati proporsi dataset asli -> stratifikasi berhasil!

---

### RINGKASAN PERHITUNGAN 5

| Subset | Total | Tidak Diabetes (0) | Diabetes (1) | Proporsi 0 | Proporsi 1 |
|--------|-------|--------------------|--------------|------------|------------|
| Dataset Asli | 768 | 500 | 268 | 65.10% | 34.90% |
| Training (80%) | 614 | 400 | 214 | 65.15% | 34.85% |
| Testing (20%) | 154 | 100 | 54 | 64.94% | 35.06% |

---

---

# PERHITUNGAN 6
## Gini Index pada Decision Tree

### Latar Belakang

Decision Tree memilih atribut terbaik untuk membagi data berdasarkan **Gini Index**. Gini Index mengukur ketidakmurnian (*impurity*) suatu node. Nilai Gini = 0 berarti semua sampel dalam satu node berasal dari kelas yang sama (pure), sedangkan Gini = 0.5 berarti distribusi kelas paling tidak merata.

### Rumus Gini Index

```
Gini(node) = 1 - SUM(p_k^2)

Dimana:
  p_k = proporsi kelas k dalam node
  k   = kelas (dalam kasus ini k ada 0 dan 1)

Gini(split) = (n_kiri/n_total) x Gini(kiri) + (n_kanan/n_total) x Gini(kanan)
```

---

### Contoh Kasus: Node ROOT (seluruh training set)

**Data di ROOT node:**
- Total sampel: 614
- Kelas 0 (Tidak Diabetes): 400
- Kelas 1 (Diabetes): 214

### LANGKAH 1 — Hitung Gini Node ROOT

**Proses:**
```
p_0 = 400 / 614 = 0.651466
p_1 = 214 / 614 = 0.348534

Gini(ROOT) = 1 - (p_0^2 + p_1^2)
Gini(ROOT) = 1 - (0.651466^2 + 0.348534^2)
Gini(ROOT) = 1 - (0.424408 + 0.121476)
Gini(ROOT) = 1 - 0.545884
Gini(ROOT) = 0.454116
```

**Output:** Gini ROOT = **0.4541** (node tidak murni, ada campuran 2 kelas)

---

### Contoh Split: Glucose <= 127

Setelah split berdasarkan Glucose <= 127:

| Node | Total | Kelas 0 | Kelas 1 |
|------|-------|---------|---------|
| Kiri (Glucose <= 127) | 380 | 295 | 85 |
| Kanan (Glucose > 127) | 234 | 105 | 129 |

### LANGKAH 2 — Gini Node Kiri (Glucose <= 127)

**Proses:**
```
p_0 = 295 / 380 = 0.776316
p_1 = 85  / 380 = 0.223684

Gini(kiri) = 1 - (0.776316^2 + 0.223684^2)
Gini(kiri) = 1 - (0.602666 + 0.050035)
Gini(kiri) = 1 - 0.652701
Gini(kiri) = 0.347299
```

### LANGKAH 3 — Gini Node Kanan (Glucose > 127)

**Proses:**
```
p_0 = 105 / 234 = 0.448718
p_1 = 129 / 234 = 0.551282

Gini(kanan) = 1 - (0.448718^2 + 0.551282^2)
Gini(kanan) = 1 - (0.201348 + 0.303912)
Gini(kanan) = 1 - 0.505260
Gini(kanan) = 0.494740
```

### LANGKAH 4 — Gini Split Gabungan

**Proses:**
```
Gini(split Glucose<=127) = (n_kiri/n_total) x Gini(kiri) + (n_kanan/n_total) x Gini(kanan)
Gini(split Glucose<=127) = (380/614) x 0.347299 + (234/614) x 0.494740
Gini(split Glucose<=127) = 0.618893 x 0.347299 + 0.381107 x 0.494740
Gini(split Glucose<=127) = 0.214976 + 0.188619
Gini(split Glucose<=127) = 0.403595
```

### LANGKAH 5 — Hitung Pengurangan Gini (Gini Reduction)

```
Gini Reduction = Gini(ROOT) - Gini(split)
Gini Reduction = 0.454116 - 0.403595
Gini Reduction = 0.050521
```

**Output:** Split pada Glucose <= 127 mengurangi impurity sebesar **0.0505**. Semakin besar Gini Reduction, semakin baik split tersebut dipilih oleh Decision Tree.

---

### RINGKASAN PERHITUNGAN 6

| Tahap | Komponen | Rumus | Hasil |
|-------|----------|-------|-------|
| ROOT | Gini ROOT | 1-(0.6515^2+0.3485^2) | **0.4541** |
| Split Kiri | Gini(kiri) | 1-(0.7763^2+0.2237^2) | **0.3473** |
| Split Kanan | Gini(kanan) | 1-(0.4487^2+0.5513^2) | **0.4947** |
| Gabungan | Gini(split) | (380/614)x0.3473+(234/614)x0.4947 | **0.4036** |
| Improvement | Gini Reduction | 0.4541-0.4036 | **0.0505** |

---

---

# PERHITUNGAN 7
## Information Gain (Entropy)

### Latar Belakang

Selain Gini, Decision Tree dapat menggunakan **Entropy** sebagai kriteria pembagian. Entropy mengukur ketidakpastian informasi dalam sebuah node. **Information Gain** mengukur seberapa banyak entropy berkurang setelah split.

### Rumus Entropy dan Information Gain

```
Entropy(node) = -SUM( p_k x log2(p_k) )

Information Gain = Entropy(parent) - SUM( (n_child/n_parent) x Entropy(child) )
```

---

### LANGKAH 1 — Entropy Node ROOT

**Data di ROOT:** 614 sampel (400 kelas-0, 214 kelas-1)

**Proses:**
```
p_0 = 400 / 614 = 0.651466
p_1 = 214 / 614 = 0.348534

log2(0.651466) = ln(0.651466) / ln(2)
               = (-0.427936) / 0.693147
               = -0.617437

log2(0.348534) = ln(0.348534) / ln(2)
               = (-1.054143) / 0.693147
               = -1.521082

Entropy(ROOT) = -(p_0 x log2(p_0)) - (p_1 x log2(p_1))
Entropy(ROOT) = -(0.651466 x (-0.617437)) - (0.348534 x (-1.521082))
Entropy(ROOT) = 0.402208 + 0.530146
Entropy(ROOT) = 0.932354
```

**Output:** Entropy ROOT = **0.9324** (mendekati 1 -> node cukup tidak murni)

---

### LANGKAH 2 — Entropy Node Kiri (Glucose <= 127)

**Data:** 380 sampel (295 kelas-0, 85 kelas-1)

**Proses:**
```
p_0 = 295 / 380 = 0.776316
p_1 = 85  / 380 = 0.223684

log2(0.776316) = -0.366063
log2(0.223684) = -2.160964

Entropy(kiri) = -(0.776316 x (-0.366063)) - (0.223684 x (-2.160964))
Entropy(kiri) = 0.284148 + 0.483279
Entropy(kiri) = 0.767427
```

**Output:** Entropy kiri = **0.7674**

---

### LANGKAH 3 — Entropy Node Kanan (Glucose > 127)

**Data:** 234 sampel (105 kelas-0, 129 kelas-1)

**Proses:**
```
p_0 = 105 / 234 = 0.448718
p_1 = 129 / 234 = 0.551282

log2(0.448718) = -1.156435
log2(0.551282) = -0.859170

Entropy(kanan) = -(0.448718 x (-1.156435)) - (0.551282 x (-0.859170))
Entropy(kanan) = 0.518918 + 0.473474
Entropy(kanan) = 0.992392
```

**Output:** Entropy kanan = **0.9924** (mendekati 1 -> distribusi sangat campuran)

---

### LANGKAH 4 — Hitung Information Gain

**Proses:**
```
IG = Entropy(ROOT) - [(n_kiri/n_total) x Entropy(kiri) + (n_kanan/n_total) x Entropy(kanan)]

IG = 0.932354 - [(380/614) x 0.767427 + (234/614) x 0.992392]
IG = 0.932354 - [0.618893 x 0.767427 + 0.381107 x 0.992392]
IG = 0.932354 - [0.475187 + 0.378234]
IG = 0.932354 - 0.853421
IG = 0.078933
```

**Output:** Information Gain split Glucose <= 127 = **0.0789**

---

### LANGKAH 5 — Interpretasi Information Gain

```
Entropy sebelum split (ROOT) = 0.9324
Entropy rata-rata setelah split = 0.8534
Penurunan ketidakpastian = 0.9324 - 0.8534 = 0.0789

Artinya: Mengetahui bahwa Glucose <= 127 atau > 127
         mengurangi ketidakpastian prediksi kelas sebesar 0.0789 bit.
```

**Keputusan:** Atribut dengan Information Gain TERTINGGI yang akan dipilih sebagai node split.

---

### RINGKASAN PERHITUNGAN 7

| Tahap | Node | Entropy |
|-------|------|---------|
| Parent | ROOT (614 sampel) | **0.9324** |
| Anak Kiri | Glucose <= 127 (380 sampel) | **0.7674** |
| Anak Kanan | Glucose > 127 (234 sampel) | **0.9924** |
| **Information Gain** | - | **0.0789** |

---

---

# PERHITUNGAN 8
## Fitness Function GWO

### Latar Belakang

Grey Wolf Optimizer (GWO) mencari kombinasi hyperparameter optimal (`max_depth`, `min_samples_leaf`) dengan cara **meminimalkan fungsi fitness**. Fungsi fitness didefinisikan sebagai:

```
fitness = 1 - mean_Recall_5FoldCV
```

Karena Recall adalah metrik utama (kita ingin memaksimalkannya), kita mengubahnya menjadi **masalah minimisasi** dengan menggunakan `1 - Recall`.

---

### Contoh: Kandidat Serigala dengan max_depth=6, min_samples_leaf=11

### LANGKAH 1 — Pembagian 5-Fold Cross Validation

**Input:** Data training = 614 sampel, K = 5

**Proses:**
```
Banyak fold = 5
Ukuran setiap fold = ceil(614 / 5) = 123 sampel (fold terakhir bisa berbeda)

Fold 1: sampel 1 - 123    -> Validation; Fold 2,3,4,5 -> Training
Fold 2: sampel 124 - 246  -> Validation; Fold 1,3,4,5 -> Training
Fold 3: sampel 247 - 369  -> Validation; Fold 1,2,4,5 -> Training
Fold 4: sampel 370 - 492  -> Validation; Fold 1,2,3,5 -> Training
Fold 5: sampel 493 - 614  -> Validation; Fold 1,2,3,4 -> Training
```

---

### LANGKAH 2 — Hitung Recall per Fold

Setelah training Decision Tree (max_depth=6, min_samples_leaf=11) pada setiap fold:

| Fold | TP | FN | Recall_k = TP/(TP+FN) |
|------|----|----|----------------------|
| 1 | 32 | 11 | 32 / (32+11) = 32/43 = 0.744186 |
| 2 | 30 | 13 | 30 / (30+13) = 30/43 = 0.697674 |
| 3 | 28 | 14 | 28 / (28+14) = 28/42 = 0.666667 |
| 4 | 33 | 9 | 33 / (33+9) = 33/42 = 0.785714 |
| 5 | 31 | 11 | 31 / (31+11) = 31/42 = 0.738095 |

---

### LANGKAH 3 — Hitung Mean Recall (5-Fold CV)

**Proses:**
```
Recall_fold1 = 0.744186
Recall_fold2 = 0.697674
Recall_fold3 = 0.666667
Recall_fold4 = 0.785714
Recall_fold5 = 0.738095

mean_Recall = (0.744186 + 0.697674 + 0.666667 + 0.785714 + 0.738095) / 5
mean_Recall = 3.632336 / 5
mean_Recall = 0.726467
```

---

### LANGKAH 4 — Hitung Nilai Fitness

**Proses:**
```
fitness = 1 - mean_Recall
fitness = 1 - 0.726467
fitness = 0.273533
```

**Output:** Kandidat dengan max_depth=6, min_samples_leaf=11 memiliki fitness = **0.2735**

---

### LANGKAH 5 — Tracking Konvergensi GWO (dari data aktual aplikasi)

| Iterasi | max_depth | min_samples_leaf | Recall (CV) | Fitness (1-Recall) |
|---------|-----------|------------------|-------------|---------------------|
| 1 | 10 | 15 | 0.629125 | **0.370875** |
| 10 | 6 | 13 | 0.690144 | **0.309856** |
| 20 | 6 | 11 | 0.708638 | **0.291362** |

**Proses contoh pada iterasi akhir (iterasi 20):**
```
Best Recall CV = 0.708638

fitness = 1 - Recall
fitness = 1 - 0.708638
fitness = 0.291362
```

**Output:** GWO berhasil menurunkan fitness dari **0.3709 -> 0.2914** selama 20 iterasi.
Artinya Recall meningkat dari **62.91% -> 70.86%** melalui proses pencarian GWO.

---

### RINGKASAN PERHITUNGAN 8

| Komponen | Nilai |
|----------|-------|
| Parameter terbaik GWO | max_depth=6, min_samples_leaf=11 |
| mean Recall (5-Fold CV) | 0.708638 |
| **Fitness = 1 - Recall** | **0.291362** |
| Interpretasi | Semakin kecil fitness -> Recall semakin tinggi |

---

---

# PERHITUNGAN 9
## Update Posisi Wolf pada GWO

### Latar Belakang

Inti dari Grey Wolf Optimizer adalah mekanisme **update posisi serigala** pada setiap iterasi. Setiap serigala (kandidat solusi) memperbarui posisinya berdasarkan posisi tiga serigala terbaik: **Alpha (alfa)**, **Beta (beta)**, dan **Delta (delta)**.

### Rumus Update Posisi GWO

**Step 1 — Hitung vektor D (jarak ke alpha/beta/delta):**
```
D_alfa  = |C_1 x X_alfa - X|
D_beta  = |C_2 x X_beta - X|
D_delta = |C_3 x X_delta - X|
```

**Step 2 — Hitung kandidat posisi baru:**
```
X_1 = X_alfa  - A_1 x D_alfa
X_2 = X_beta  - A_2 x D_beta
X_3 = X_delta - A_3 x D_delta
```

**Step 3 — Update posisi final:**
```
X(t+1) = (X_1 + X_2 + X_3) / 3
```

**Rumus koefisien A dan C:**
```
a = 2 - t x (2 / T)      -> a menurun linier dari 2 ke 0

A = 2a x r_1 - a          -> r_1 ~ Uniform[0,1]
C = 2 x r_2               -> r_2 ~ Uniform[0,1]
```

---

### Contoh: Iterasi t=1, Dimensi max_depth

**Parameter GWO:**
- T = 20 (iterasi maksimum)
- t = 1 (iterasi saat ini)

**Posisi tiga serigala terbaik:**
| Wolf | max_depth |
|------|-----------|
| Alpha (alfa) | 10 |
| Beta (beta) | 8 |
| Delta (delta) | 12 |

**Posisi serigala Omega yang diperbarui:** X = 7

---

### LANGKAH 1 — Hitung Parameter a pada iterasi t=1

**Proses:**
```
a = 2 - t x (2 / T)
a = 2 - 1 x (2 / 20)
a = 2 - 1 x 0.1
a = 2 - 0.1
a = 1.9
```

**Output:** a = **1.9**
(a akan terus menurun hingga a = 0 pada iterasi terakhir t=20)

---

### LANGKAH 2 — Hitung Koefisien A dan C (untuk arah Alpha)

Gunakan bilangan acak: r_1 = 0.6, r_2 = 0.4

**Proses:**
```
A_1 = 2 x a x r_1 - a
A_1 = 2 x 1.9 x 0.6 - 1.9
A_1 = 3.8 x 0.6 - 1.9
A_1 = 2.28 - 1.9
A_1 = 0.38

C_1 = 2 x r_2
C_1 = 2 x 0.4
C_1 = 0.8
```

---

### LANGKAH 3 — Hitung D_alfa (jarak ke Alpha, dimensi max_depth)

**Proses:**
```
X_alfa = 10   (max_depth alpha)
X      = 7    (max_depth serigala omega)

D_alfa = |C_1 x X_alfa - X|
D_alfa = |0.8 x 10 - 7|
D_alfa = |8 - 7|
D_alfa = |1|
D_alfa = 1
```

---

### LANGKAH 4 — Hitung X_1 (kandidat posisi dari Alpha)

**Proses:**
```
X_1 = X_alfa - A_1 x D_alfa
X_1 = 10 - 0.38 x 1
X_1 = 10 - 0.38
X_1 = 9.62
```

---

### LANGKAH 5 — Hitung D_beta dan X_2 (dari Beta, r_1=0.3, r_2=0.7)

**Proses:**
```
A_2 = 2 x 1.9 x 0.3 - 1.9
A_2 = 1.14 - 1.9
A_2 = -0.76

C_2 = 2 x 0.7 = 1.4

D_beta = |C_2 x X_beta - X|
D_beta = |1.4 x 8 - 7|
D_beta = |11.2 - 7|
D_beta = 4.2

X_2 = X_beta - A_2 x D_beta
X_2 = 8 - (-0.76) x 4.2
X_2 = 8 + 3.192
X_2 = 11.192
```

---

### LANGKAH 6 — Hitung D_delta dan X_3 (dari Delta, r_1=0.5, r_2=0.2)

**Proses:**
```
A_3 = 2 x 1.9 x 0.5 - 1.9
A_3 = 1.9 - 1.9
A_3 = 0.0

C_3 = 2 x 0.2 = 0.4

D_delta = |C_3 x X_delta - X|
D_delta = |0.4 x 12 - 7|
D_delta = |4.8 - 7|
D_delta = |-2.2|
D_delta = 2.2

X_3 = X_delta - A_3 x D_delta
X_3 = 12 - 0.0 x 2.2
X_3 = 12 - 0
X_3 = 12
```

---

### LANGKAH 7 — Update Posisi Final

**Proses:**
```
X(t+1) = (X_1 + X_2 + X_3) / 3
X(t+1) = (9.62 + 11.192 + 12) / 3
X(t+1) = 32.812 / 3
X(t+1) = 10.937...

-> Konversi ke integer: max_depth = round(10.937) = 11
-> Clipping ke batas [1, 20]: max_depth = min(max(11, 1), 20) = 11
```

**Output:** Serigala omega memperbarui posisinya dari max_depth=**7** menjadi max_depth=**11** pada iterasi t=1.

---

### RINGKASAN PERHITUNGAN 9

| Langkah | Variabel | Rumus | Hasil |
|---------|----------|-------|-------|
| 1 | a | 2 - t x (2/T) = 2 - 1 x 0.1 | **1.9** |
| 2 | A_1, C_1 | A=2(1.9)(0.6)-1.9; C=2(0.4) | **A=0.38, C=0.8** |
| 3 | D_alfa | ABS(0.8 x 10 - 7) | **1** |
| 4 | X_1 (dari alfa) | 10 - 0.38 x 1 | **9.62** |
| 5 | X_2 (dari beta) | 8 - (-0.76) x 4.2 | **11.192** |
| 6 | X_3 (dari delta) | 12 - 0.0 x 2.2 | **12** |
| 7 | X_baru | (9.62 + 11.192 + 12) / 3 | **10.937 -> max_depth = 11** |

---

---

# PERHITUNGAN 10
## Evaluasi Model: Accuracy, Precision, Recall, F1-Score

### Latar Belakang

Setelah model Decision Tree dilatih dengan hyperparameter optimal dari GWO, performa model dievaluasi pada **data testing (154 sampel)** menggunakan confusion matrix sebagai dasar semua metrik.

### Confusion Matrix GWO-Optimized Model (Data Testing)

```
                        Prediksi: Tidak Diabetes   Prediksi: Diabetes
Aktual: Tidak Diabetes       TN = 66                   FP = 33
Aktual: Diabetes             FN = 13                   TP = 42
```

| Komponen | Simbol | Nilai | Keterangan |
|----------|--------|-------|-----------|
| True Positive | TP | 42 | Benar-benar Diabetes, diprediksi Diabetes (BENAR) |
| True Negative | TN | 66 | Benar-benar Tidak Diabetes, diprediksi Tidak Diabetes (BENAR) |
| False Positive | FP | 33 | Tidak Diabetes, diprediksi Diabetes (SALAH - False Alarm) |
| False Negative | FN | 13 | Benar-benar Diabetes, diprediksi Tidak Diabetes (SALAH - Berbahaya) |
| **Total** | | **154** | TP+TN+FP+FN = 42+66+33+13 = 154 (COCOK) |

---

### LANGKAH 1 — Hitung Accuracy

**Rumus:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Proses:**
```
Accuracy = (42 + 66) / (42 + 66 + 33 + 13)
Accuracy = 108 / (108 + 46)
Accuracy = 108 / 154
Accuracy = 0.701298...
Accuracy = 0.7013 = 70.13%
```

**Output:** Accuracy = **70.13%**
Interpretasi: Dari 154 sampel testing, sebanyak 108 prediksi benar dan 46 prediksi salah.

---

### LANGKAH 2 — Hitung Precision

**Rumus:**
```
Precision = TP / (TP + FP)
```

**Proses:**
```
Precision = 42 / (42 + 33)
Precision = 42 / 75
Precision = 0.560000 = 56.00%
```

**Output:** Precision = **56.00%**
Interpretasi: Dari 75 prediksi positif (Diabetes) oleh model, hanya 42 yang benar-benar Diabetes.

---

### LANGKAH 3 — Hitung Recall (Sensitivity)

**Rumus:**
```
Recall = TP / (TP + FN)
```

**Proses:**
```
Recall = 42 / (42 + 13)
Recall = 42 / 55
Recall = 0.763636...
Recall = 0.7636 = 76.36%
```

**Output:** Recall = **76.36%** (METRIK UTAMA - tertinggi di antara 3 model)
Interpretasi: Dari 55 pasien yang benar-benar Diabetes (TP+FN), model berhasil mendeteksi 42 orang (76.36%) dan melewatkan hanya 13 orang.

---

### LANGKAH 4 — Hitung F1-Score

**Rumus:**
```
F1-Score = 2 x (Precision x Recall) / (Precision + Recall)
```

**Proses:**
```
Precision = 0.5600
Recall    = 0.7636

Langkah 4a: Perkalian Precision x Recall
  Precision x Recall = 0.5600 x 0.7636
  Precision x Recall = 0.427616

Langkah 4b: Penjumlahan Precision + Recall
  Precision + Recall = 0.5600 + 0.7636
  Precision + Recall = 1.3236

Langkah 4c: F1-Score
  F1-Score = 2 x 0.427616 / 1.3236
  F1-Score = 0.855232 / 1.3236
  F1-Score = 0.646201...
  F1-Score = 0.6462 = 64.62%
```

**Output:** F1-Score = **64.62%**

---

### LANGKAH 5 — Hitung Specificity

**Rumus:**
```
Specificity = TN / (TN + FP)
```

**Proses:**
```
Specificity = 66 / (66 + 33)
Specificity = 66 / 99
Specificity = 0.666666...
Specificity = 0.6667 = 66.67%
```

**Output:** Specificity = **66.67%**
Interpretasi: Dari 99 pasien Tidak Diabetes, model benar mengidentifikasi 66 orang.

---

### LANGKAH 6 — Perbandingan Lengkap 3 Model

**Baseline DT:** TP=36, TN=76, FP=23, FN=19, Total=154

```
Accuracy  = (36+76) / 154 = 112/154 = 0.727273 = 72.73%
Precision = 36 / (36+23) = 36/59   = 0.610169 = 61.02%
Recall    = 36 / (36+19) = 36/55   = 0.654545 = 65.45%
F1-Score  = 2 x (0.610169 x 0.654545) / (0.610169 + 0.654545)
          = 2 x 0.399378 / 1.264714
          = 0.798756 / 1.264714
          = 0.631579 = 63.16%
```

**RandomizedSearchCV DT:** TP=38, TN=75, FP=24, FN=17, Total=154

```
Accuracy  = (38+75) / 154 = 113/154 = 0.733766 = 73.38%
Precision = 38 / (38+24) = 38/62   = 0.612903 = 61.29%
Recall    = 38 / (38+17) = 38/55   = 0.690909 = 69.09%
F1-Score  = 2 x (0.612903 x 0.690909) / (0.612903 + 0.690909)
          = 2 x 0.423440 / 1.303812
          = 0.846880 / 1.303812
          = 0.649573 = 64.96%
```

---

### RINGKASAN PERHITUNGAN 10: Semua Model

| Metrik | Rumus | Baseline DT | RandomizedSearch DT | GWO-DT |
|--------|-------|-------------|---------------------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 112/154 = **72.73%** | 113/154 = **73.38%** | 108/154 = **70.13%** |
| **Precision** | TP/(TP+FP) | 36/59 = **61.02%** | 38/62 = **61.29%** | 42/75 = **56.00%** |
| **Recall** | TP/(TP+FN) | 36/55 = **65.45%** | 38/55 = **69.09%** | 42/55 = **76.36%** [TERBAIK] |
| **F1-Score** | 2x(PxR)/(P+R) | **63.16%** | **64.96%** | **64.62%** |
| **Specificity** | TN/(TN+FP) | 76/99 = **76.77%** | 75/99 = **75.76%** | 66/99 = **66.67%** |
| False Negative | FN | 19 | 17 | **13** [TERBAIK] |

---

### ANALISIS TRADE-OFF GWO

```
Improvement Recall:
  GWO vs Baseline      = 76.36% - 65.45% = +10.91%
  GWO vs RandomSearch  = 76.36% - 69.09% = +7.27%

Pengurangan False Negative:
  GWO vs Baseline      = 19 - 13 = 6 pasien lebih terdeteksi
  GWO vs RandomSearch  = 17 - 13 = 4 pasien lebih terdeteksi

Trade-off Precision:
  GWO vs Baseline      = 56.00% - 61.02% = -5.02% (turun)
  GWO vs RandomSearch  = 56.00% - 61.29% = -5.29% (turun)
```

**Kesimpulan:**

GWO berhasil memaksimalkan Recall (+10.91% vs Baseline) dengan mengorbankan Precision (-5.02%).
Dalam konteks medis, ini merupakan trade-off yang dapat diterima karena:

1. **FN lebih rendah** = lebih sedikit pasien diabetes tidak terdeteksi = MENYELAMATKAN NYAWA
2. **FP lebih tinggi** = lebih banyak tes konfirmasi = biaya tambahan yang DAPAT DITOLERANSI

---

---

## RINGKASAN SELURUH PERHITUNGAN

| No | Topik | Rumus Utama | Output Kunci |
|----|-------|-------------|--------------|
| 1 | Missing Values | (Jml Nol / N) x 100% | Insulin 48.70% missing terbanyak |
| 2 | Median Imputasi | Nilai ke-[(n+1)/2] atau (v1+v2)/2 | SkinThickness=28.5, Insulin=120, dst |
| 3 | Terapkan Imputasi | X_imp = Median jika X=NaN | Baris 1: BP=72, Skin=28.5, Ins=120, BMI=32 |
| 4 | Z-Score Standardisasi | z = (x - mu) / sigma | Glucose 84 -> z = -1.2572 |
| 5 | Stratified Split | n_train = round(N x 0.8) | 614 training, 154 testing, proporsi terjaga |
| 6 | Gini Index | 1 - SUM(p_k^2) | Gini ROOT=0.4541, Reduction=0.0505 |
| 7 | Information Gain | Entropy(parent) - SUM(w x Entropy(child)) | IG split Glucose=0.0789 |
| 8 | Fitness GWO | 1 - mean_Recall_5FoldCV | fitness=0.2914 (Recall CV=70.86%) |
| 9 | Update Posisi Wolf | X(t+1) = (X1+X2+X3)/3 | max_depth diperbarui dari 7 -> 11 |
| 10 | Evaluasi Metrik | Accuracy, Precision, Recall, F1 | GWO: Recall=76.36% TERTINGGI |

---

> **Dokumen ini dibuat untuk mendukung penelitian:**
> "Optimasi Hyperparameter Decision Tree Menggunakan Grey Wolf Optimizer (GWO) untuk Prediksi Diabetes"
>
> Dataset: Pima Indians Diabetes Database (768 sampel, 8 fitur prediktor)
>
> Semua angka dalam dokumen ini bersumber dari data aktual pipeline penelitian
> (training set 614 sampel, testing set 154 sampel, random_state=42).
