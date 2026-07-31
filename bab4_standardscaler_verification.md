# Verifikasi StandardScaler Pipeline Utama Bab 4

Pipeline yang digunakan sama dengan preprocessing utama project pada `model_utils.py`: nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diubah menjadi NaN; data dibagi train-test 80:20 dengan `random_state=42`; median di-fit dari data training; lalu `StandardScaler` di-fit pada data training dan digunakan untuk transformasi data training serta testing.

Tidak ada penggunaan MinMaxScaler pada verifikasi ini.

## Jumlah Data

| Subset | Jumlah Data |
| --- | --- |
| Training | 614 |
| Testing | 154 |

## Mean dan Standar Deviasi StandardScaler

Nilai mean (μ) dan standar deviasi (σ) berikut berasal dari objek `StandardScaler` setelah `fit` pada data training hasil imputasi median.

| Atribut | Mean (μ) | Standar Deviasi (σ) |
| --- | --- | --- |
| Pregnancies | 3.742671 | 3.310565 |
| Glucose | 121.815961 | 30.079488 |
| BloodPressure | 72.229642 | 12.089421 |
| SkinThickness | 28.568404 | 8.403178 |
| Insulin | 138.115635 | 88.578061 |
| BMI | 32.348208 | 6.929968 |
| DiabetesPedigreeFunction | 0.469168 | 0.336572 |
| Age | 32.907166 | 11.494065 |

## 10 Baris Pertama Data Training Setelah Imputasi Median Sebelum StandardScaler

| No | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.0000 | 84.0000 | 72.0000 | 28.5000 | 120.0000 | 32.0000 | 0.3040 | 21.0000 |
| 2 | 9.0000 | 112.0000 | 82.0000 | 24.0000 | 120.0000 | 28.2000 | 1.2820 | 50.0000 |
| 3 | 1.0000 | 139.0000 | 46.0000 | 19.0000 | 83.0000 | 28.7000 | 0.6540 | 22.0000 |
| 4 | 0.0000 | 161.0000 | 50.0000 | 28.5000 | 120.0000 | 21.9000 | 0.2540 | 65.0000 |
| 5 | 6.0000 | 134.0000 | 80.0000 | 37.0000 | 370.0000 | 46.2000 | 0.2380 | 46.0000 |
| 6 | 1.0000 | 130.0000 | 70.0000 | 13.0000 | 105.0000 | 25.9000 | 0.4720 | 22.0000 |
| 7 | 4.0000 | 132.0000 | 72.0000 | 28.5000 | 120.0000 | 32.9000 | 0.3020 | 23.0000 |
| 8 | 10.0000 | 161.0000 | 68.0000 | 23.0000 | 132.0000 | 25.5000 | 0.3260 | 47.0000 |
| 9 | 1.0000 | 108.0000 | 60.0000 | 46.0000 | 178.0000 | 35.5000 | 0.4150 | 24.0000 |
| 10 | 1.0000 | 80.0000 | 55.0000 | 28.5000 | 120.0000 | 19.1000 | 0.2580 | 21.0000 |

## 10 Baris Pertama Data Training Setelah StandardScaler

| No | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -0.5264 | -1.2572 | -0.0190 | -0.0081 | -0.2045 | -0.0502 | -0.4907 | -1.0359 |
| 2 | 1.5880 | -0.3263 | 0.8082 | -0.5437 | -0.2045 | -0.5986 | 2.4150 | 1.4871 |
| 3 | -0.8285 | 0.5713 | -2.1696 | -1.1387 | -0.6222 | -0.5264 | 0.5492 | -0.9489 |
| 4 | -1.1305 | 1.3027 | -1.8388 | -0.0081 | -0.2045 | -1.5077 | -0.6393 | 2.7921 |
| 5 | 0.6819 | 0.4051 | 0.6427 | 1.0034 | 2.6179 | 1.9988 | -0.6868 | 1.1391 |
| 6 | -0.8285 | 0.2721 | -0.1844 | -1.8527 | -0.3739 | -0.9305 | 0.0084 | -0.9489 |
| 7 | 0.0777 | 0.3386 | -0.0190 | -0.0081 | -0.2045 | 0.0796 | -0.4967 | -0.8619 |
| 8 | 1.8901 | 1.3027 | -0.3499 | -0.6627 | -0.0690 | -0.9882 | -0.4254 | 1.2261 |
| 9 | -0.8285 | -0.4593 | -1.0116 | 2.0744 | 0.4503 | 0.4548 | -0.1609 | -0.7749 |
| 10 | -0.8285 | -1.3902 | -1.4252 | -0.0081 | -0.2045 | -1.9117 | -0.6274 | -1.0359 |

## Rumus StandardScaler

```text
z = (x - μ) / σ
```

## Verifikasi Perhitungan Manual Baris Pertama

### Contoh Glucose

Nilai asli: Glucose = 84
Mean: μ = 121.815961
Standar deviasi: σ = 30.079488

Rumus:
```text
z = (x - μ) / σ
z = (84 - 121.815961) / 30.079488
z = -37.815961 / 30.079488
z = -1.257201
```
Hasil program StandardScaler pada baris pertama atribut Glucose = -1.257201.
Kesimpulan contoh Glucose: hasil manual sama dengan output program.

### Contoh BloodPressure

Nilai asli: BloodPressure = 72
Mean: μ = 72.229642
Standar deviasi: σ = 12.089421

Rumus:
```text
z = (x - μ) / σ
z = (72 - 72.229642) / 12.089421
z = -0.229642 / 12.089421
z = -0.018995
```
Hasil program StandardScaler pada baris pertama atribut BloodPressure = -0.018995.
Kesimpulan contoh BloodPressure: hasil manual sama dengan output program.

### Contoh SkinThickness

Nilai asli: SkinThickness = 28.500000
Mean: μ = 28.568404
Standar deviasi: σ = 8.403178

Rumus:
```text
z = (x - μ) / σ
z = (28.500000 - 28.568404) / 8.403178
z = -0.068404 / 8.403178
z = -0.008140
```
Hasil program StandardScaler pada baris pertama atribut SkinThickness = -0.008140.
Kesimpulan contoh SkinThickness: hasil manual sama dengan output program.

## Tabel Verifikasi Manual

| Atribut | Nilai Asli Baris 1 (x) | Mean (μ) | Standar Deviasi (σ) | Rumus | Substitusi | Hasil Manual | Hasil StandardScaler | Sama |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Glucose | 84 | 121.815961 | 30.079488 | z = (x - μ) / σ | z = (84 - 121.815961) / 30.079488 | -1.257201 | -1.257201 | Ya |
| BloodPressure | 72 | 72.229642 | 12.089421 | z = (x - μ) / σ | z = (72 - 72.229642) / 12.089421 | -0.018995 | -0.018995 | Ya |
| SkinThickness | 28.500000 | 28.568404 | 8.403178 | z = (x - μ) / σ | z = (28.500000 - 28.568404) / 8.403178 | -0.008140 | -0.008140 | Ya |

## Kesimpulan

- StandardScaler berhasil diterapkan pada data training dan data testing menggunakan parameter mean dan standar deviasi yang di-fit dari data training.
- Hasil perhitungan manual untuk Glucose, BloodPressure, dan SkinThickness sama dengan hasil transformasi program StandardScaler pada baris pertama data training.
- Dengan demikian, proses standardisasi pada eksperimen Bab 4 konsisten dengan pipeline utama aplikasi.