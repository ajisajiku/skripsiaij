# Verifikasi Median Data Training Bab 4

Pipeline yang diverifikasi:
1. Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diubah menjadi NaN.
2. Dataset dibagi menjadi data training dan testing dengan rasio 80:20 menggunakan random_state=42.
3. Median dihitung hanya dari data training, dengan NaN diabaikan.
4. Median dibandingkan dengan hasil SimpleImputer(strategy="median").

Jumlah data training: 614
Jumlah data testing: 154

Catatan penting:
- Jumlah data training tetap 614, tetapi jumlah data valid untuk Glucose adalah 609 karena nilai 0 pada atribut tersebut diubah menjadi NaN dan diabaikan dalam perhitungan median.
- Jumlah data training tetap 614, tetapi jumlah data valid untuk BloodPressure adalah 590 karena nilai 0 pada atribut tersebut diubah menjadi NaN dan diabaikan dalam perhitungan median.
- Jumlah data training tetap 614, tetapi jumlah data valid untuk SkinThickness adalah 438 karena nilai 0 pada atribut tersebut diubah menjadi NaN dan diabaikan dalam perhitungan median.
- Jumlah data training tetap 614, tetapi jumlah data valid untuk Insulin adalah 324 karena nilai 0 pada atribut tersebut diubah menjadi NaN dan diabaikan dalam perhitungan median.
- Jumlah data training tetap 614, tetapi jumlah data valid untuk BMI adalah 607 karena nilai 0 pada atribut tersebut diubah menjadi NaN dan diabaikan dalam perhitungan median.

## Tabel Verifikasi Median

| Atribut | Jumlah Data Valid | Posisi Tengah 1 | Nilai Tengah 1 | Posisi Tengah 2 | Nilai Tengah 2 | Perhitungan Median | Median SimpleImputer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Glucose | 609 | 305 | 118 |  |  | Median = nilai ke-305 = 118 | 118 |
| BloodPressure | 590 | 295 | 72 | 296 | 72 | Median = (72 + 72) / 2 = 72 | 72 |
| SkinThickness | 438 | 219 | 28 | 220 | 29 | Median = (28 + 29) / 2 = 28.5000 | 28.5000 |
| Insulin | 324 | 162 | 120 | 163 | 120 | Median = (120 + 120) / 2 = 120 | 120 |
| BMI | 607 | 304 | 32 |  |  | Median = nilai ke-304 = 32 | 32 |

## Validasi SimpleImputer

| Atribut | Ganjil/Genap | Median SimpleImputer | Sesuai SimpleImputer |
| --- | --- | --- | --- |
| Glucose | Ganjil | 118 | Ya |
| BloodPressure | Genap | 72 | Ya |
| SkinThickness | Genap | 28.5000 | Ya |
| Insulin | Genap | 120 | Ya |
| BMI | Ganjil | 32 | Ya |

Kesimpulan: seluruh median manual sama dengan hasil SimpleImputer(strategy="median"), sehingga nilai median Bab 4 sudah sesuai dengan pipeline utama.