# Verifikasi Decision Tree Baseline Bab 4

Pipeline yang digunakan sama dengan aplikasi utama: train-test split 80:20 dengan `random_state=42`, nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diubah menjadi NaN, imputasi median berdasarkan data training, StandardScaler di-fit pada data training, dan Decision Tree baseline dilatih tanpa optimasi hyperparameter.

## Jumlah Data

| Subset | Jumlah Data |
| --- | --- |
| Training | 614 |
| Testing | 154 |

## Parameter Decision Tree Baseline

| Parameter | Nilai |
| --- | --- |
| ccp_alpha | 0.0 |
| class_weight | None |
| criterion | gini |
| max_depth | None |
| max_features | None |
| max_leaf_nodes | None |
| min_impurity_decrease | 0.0 |
| min_samples_leaf | 1 |
| min_samples_split | 2 |
| min_weight_fraction_leaf | 0.0 |
| monotonic_cst | None |
| random_state | 42 |
| splitter | best |

## Confusion Matrix Baseline

| Komponen | Jumlah | Keterangan |
| --- | --- | --- |
| TN | 76 | Aktual Tidak Diabetes, prediksi Tidak Diabetes |
| FP | 23 | Aktual Tidak Diabetes, prediksi Diabetes |
| FN | 19 | Aktual Diabetes, prediksi Tidak Diabetes |
| TP | 36 | Aktual Diabetes, prediksi Diabetes |

## Metrik Evaluasi Program dan Manual

| Metrik | Hasil Program | Hasil Manual | Sama |
| --- | --- | --- | --- |
| Accuracy | 0.727273 | 0.727273 | Ya |
| Precision | 0.610169 | 0.610169 | Ya |
| Recall | 0.654545 | 0.654545 | Ya |
| F1-Score | 0.631579 | 0.631579 | Ya |

## 10 Data Testing Pertama dan Prediksi Baseline

| No | Index Dataset | Outcome Aktual | Prediksi Baseline | Status |
| --- | --- | --- | --- | --- |
| 1 | 668 | Tidak Diabetes | Tidak Diabetes | Benar |
| 2 | 324 | Tidak Diabetes | Tidak Diabetes | Benar |
| 3 | 624 | Tidak Diabetes | Tidak Diabetes | Benar |
| 4 | 690 | Tidak Diabetes | Tidak Diabetes | Benar |
| 5 | 473 | Tidak Diabetes | Tidak Diabetes | Benar |
| 6 | 204 | Tidak Diabetes | Diabetes | Salah |
| 7 | 97 | Tidak Diabetes | Tidak Diabetes | Benar |
| 8 | 336 | Tidak Diabetes | Diabetes | Salah |
| 9 | 568 | Tidak Diabetes | Diabetes | Salah |
| 10 | 148 | Tidak Diabetes | Diabetes | Salah |

## Contoh Data Berdasarkan Kategori Confusion Matrix

| Kategori | No Testing | Index Dataset | Outcome Aktual | Prediksi Baseline | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TP | 19 | 604 | Diabetes | Diabetes | 4.0000 | 183.0000 | NaN | NaN | NaN | 28.4000 | 0.2120 | 36.0000 |
| TP | 20 | 213 | Diabetes | Diabetes | 0.0000 | 140.0000 | 65.0000 | 26.0000 | 130.0000 | 42.6000 | 0.4310 | 24.0000 |
| TP | 26 | 209 | Diabetes | Diabetes | 7.0000 | 184.0000 | 84.0000 | 33.0000 | NaN | 35.5000 | 0.3550 | 41.0000 |
| TP | 31 | 120 | Diabetes | Diabetes | 0.0000 | 162.0000 | 76.0000 | 56.0000 | 100.0000 | 53.2000 | 0.7590 | 25.0000 |
| TP | 32 | 363 | Diabetes | Diabetes | 4.0000 | 146.0000 | 78.0000 | NaN | NaN | 38.5000 | 0.5200 | 67.0000 |
| TN | 1 | 668 | Tidak Diabetes | Tidak Diabetes | 6.0000 | 98.0000 | 58.0000 | 33.0000 | 190.0000 | 34.0000 | 0.4300 | 43.0000 |
| TN | 2 | 324 | Tidak Diabetes | Tidak Diabetes | 2.0000 | 112.0000 | 75.0000 | 32.0000 | NaN | 35.7000 | 0.1480 | 21.0000 |
| TN | 3 | 624 | Tidak Diabetes | Tidak Diabetes | 2.0000 | 108.0000 | 64.0000 | NaN | NaN | 30.8000 | 0.1580 | 21.0000 |
| TN | 4 | 690 | Tidak Diabetes | Tidak Diabetes | 8.0000 | 107.0000 | 80.0000 | NaN | NaN | 24.6000 | 0.8560 | 34.0000 |
| TN | 5 | 473 | Tidak Diabetes | Tidak Diabetes | 7.0000 | 136.0000 | 90.0000 | NaN | NaN | 29.9000 | 0.2100 | 50.0000 |
| FP | 6 | 204 | Tidak Diabetes | Diabetes | 6.0000 | 103.0000 | 72.0000 | 32.0000 | 190.0000 | 37.7000 | 0.3240 | 55.0000 |
| FP | 8 | 336 | Tidak Diabetes | Diabetes | 0.0000 | 117.0000 | NaN | NaN | NaN | 33.8000 | 0.9320 | 44.0000 |
| FP | 9 | 568 | Tidak Diabetes | Diabetes | 4.0000 | 154.0000 | 72.0000 | 29.0000 | 126.0000 | 31.3000 | 0.3380 | 37.0000 |
| FP | 10 | 148 | Tidak Diabetes | Diabetes | 5.0000 | 147.0000 | 78.0000 | NaN | NaN | 33.7000 | 0.2180 | 65.0000 |
| FP | 12 | 212 | Tidak Diabetes | Diabetes | 7.0000 | 179.0000 | 95.0000 | 31.0000 | NaN | 34.2000 | 0.1640 | 60.0000 |
| FN | 11 | 667 | Diabetes | Tidak Diabetes | 10.0000 | 111.0000 | 70.0000 | 27.0000 | NaN | 27.5000 | 0.1410 | 40.0000 |
| FN | 13 | 199 | Diabetes | Tidak Diabetes | 4.0000 | 148.0000 | 60.0000 | 27.0000 | 318.0000 | 30.9000 | 0.1500 | 29.0000 |
| FN | 16 | 356 | Diabetes | Tidak Diabetes | 1.0000 | 125.0000 | 50.0000 | 40.0000 | 167.0000 | 33.3000 | 0.9620 | 28.0000 |
| FN | 29 | 328 | Diabetes | Tidak Diabetes | 2.0000 | 102.0000 | 86.0000 | 36.0000 | 120.0000 | 45.5000 | 0.1270 | 23.0000 |
| FN | 38 | 66 | Diabetes | Tidak Diabetes | 0.0000 | 109.0000 | 88.0000 | 30.0000 | NaN | 32.5000 | 0.8550 | 38.0000 |

## Perhitungan Manual Metrik

| Metrik | Rumus | Substitusi | Perhitungan | Hasil |
| --- | --- | --- | --- | --- |
| Accuracy | Accuracy = (TP + TN) / (TP + TN + FP + FN) | (36 + 76) / (36 + 76 + 23 + 19) | 112 / 154 | 0.727273 |
| Precision | Precision = TP / (TP + FP) | 36 / (36 + 23) | 36 / 59 | 0.610169 |
| Recall | Recall = TP / (TP + FN) | 36 / (36 + 19) | 36 / 55 | 0.654545 |
| F1-Score | F1 = 2 x (Precision x Recall) / (Precision + Recall) | 2 x (0.610169 x 0.654545) / (0.610169 + 0.654545) | 0.798767 / 1.264715 | 0.631579 |

### Langkah Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy = (36 + 76) / (36 + 76 + 23 + 19)
Accuracy = 112 / 154
Accuracy = 0.727273
```

### Langkah Precision

```text
Precision = TP / (TP + FP)
Precision = 36 / (36 + 23)
Precision = 36 / 59
Precision = 0.610169
```

### Langkah Recall

```text
Recall = TP / (TP + FN)
Recall = 36 / (36 + 19)
Recall = 36 / 55
Recall = 0.654545
```

### Langkah F1-Score

```text
F1 = 2 x (Precision x Recall) / (Precision + Recall)
F1 = 2 x (0.610169 x 0.654545) / (0.610169 + 0.654545)
F1 = 0.798767 / 1.264715
F1 = 0.631579
```

## Kesimpulan

- Perhitungan manual Accuracy, Precision, Recall, dan F1-Score sama dengan output program.
- Jumlah false negative (FN) adalah 19. Artinya terdapat 19 data pasien diabetes yang diprediksi tidak diabetes oleh model baseline.
- Jumlah false positive (FP) adalah 23. Artinya terdapat 23 data pasien tidak diabetes yang diprediksi diabetes oleh model baseline.
- Dalam konteks deteksi diabetes, false negative perlu diperhatikan karena pasien yang sebenarnya diabetes dapat tidak terdeteksi oleh model.