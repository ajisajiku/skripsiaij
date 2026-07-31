# Verifikasi Hasil Imputasi Median Data Training Bab 4

Pipeline yang digunakan:
1. Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diubah menjadi NaN.
2. Dataset dibagi menjadi data training dan testing dengan train-test split 80:20 menggunakan random_state=42.
3. Median dihitung dari data training saja menggunakan SimpleImputer(strategy="median").
4. Median diterapkan untuk mengganti NaN pada data training.

Jumlah data training: 614
Jumlah data testing: 154

Keterangan tanda: nilai dengan format `NaN -> nilai *` menunjukkan kolom yang berubah akibat imputasi median.

## Jumlah NaN Sebelum dan Sesudah Imputasi

| Atribut | NaN Sebelum Imputasi | Median Training | NaN Setelah Imputasi |
| --- | --- | --- | --- |
| Glucose | 5 | 118 | 0 |
| BloodPressure | 24 | 72 | 0 |
| SkinThickness | 176 | 28.5000 | 0 |
| Insulin | 290 | 120 | 0 |
| BMI | 7 | 32 | 0 |

## 10 Baris Pertama Data Training Sebelum Imputasi

| No | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 84 | NaN | NaN | NaN | NaN | 0.3040 | 21 |
| 2 | 9 | 112 | 82 | 24 | NaN | 28.2000 | 1.2820 | 50 |
| 3 | 1 | 139 | 46 | 19 | 83 | 28.7000 | 0.6540 | 22 |
| 4 | 0 | 161 | 50 | NaN | NaN | 21.9000 | 0.2540 | 65 |
| 5 | 6 | 134 | 80 | 37 | 370 | 46.2000 | 0.2380 | 46 |
| 6 | 1 | 130 | 70 | 13 | 105 | 25.9000 | 0.4720 | 22 |
| 7 | 4 | 132 | NaN | NaN | NaN | 32.9000 | 0.3020 | 23 |
| 8 | 10 | 161 | 68 | 23 | 132 | 25.5000 | 0.3260 | 47 |
| 9 | 1 | 108 | 60 | 46 | 178 | 35.5000 | 0.4150 | 24 |
| 10 | 1 | 80 | 55 | NaN | NaN | 19.1000 | 0.2580 | 21 |

## 10 Baris Pertama Data Training Sesudah Imputasi dan Penanda Perubahan

| No | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Kolom Berubah Akibat Imputasi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 84 | NaN -> 72 * | NaN -> 28.5000 * | NaN -> 120 * | NaN -> 32 * | 0.3040 | 21 | BloodPressure, SkinThickness, Insulin, BMI |
| 2 | 9 | 112 | 82 | 24 | NaN -> 120 * | 28.2000 | 1.2820 | 50 | Insulin |
| 3 | 1 | 139 | 46 | 19 | 83 | 28.7000 | 0.6540 | 22 | - |
| 4 | 0 | 161 | 50 | NaN -> 28.5000 * | NaN -> 120 * | 21.9000 | 0.2540 | 65 | SkinThickness, Insulin |
| 5 | 6 | 134 | 80 | 37 | 370 | 46.2000 | 0.2380 | 46 | - |
| 6 | 1 | 130 | 70 | 13 | 105 | 25.9000 | 0.4720 | 22 | - |
| 7 | 4 | 132 | NaN -> 72 * | NaN -> 28.5000 * | NaN -> 120 * | 32.9000 | 0.3020 | 23 | BloodPressure, SkinThickness, Insulin |
| 8 | 10 | 161 | 68 | 23 | 132 | 25.5000 | 0.3260 | 47 | - |
| 9 | 1 | 108 | 60 | 46 | 178 | 35.5000 | 0.4150 | 24 | - |
| 10 | 1 | 80 | 55 | NaN -> 28.5000 * | NaN -> 120 * | 19.1000 | 0.2580 | 21 | SkinThickness, Insulin |

## Ringkasan Kolom yang Berubah pada 10 Baris Pertama

| Atribut | Jumlah Perubahan pada 10 Baris Pertama | Nilai Imputasi |
| --- | --- | --- |
| BloodPressure | 2 | 72 |
| SkinThickness | 4 | 28.5000 |
| Insulin | 5 | 120 |
| BMI | 1 | 32 |

Narasi akademik singkat:
Hasil verifikasi menunjukkan bahwa sebelum imputasi masih terdapat NaN pada atribut tertentu di data training. Setelah SimpleImputer(strategy="median") diterapkan, jumlah NaN pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI menjadi 0. Pada 10 baris pertama data training, perubahan terlihat pada sel yang awalnya NaN dan kemudian terisi nilai median dari data training, misalnya SkinThickness dan Insulin diisi dengan median training masing-masing.