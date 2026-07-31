# Verifikasi Model Final GWO Tersimpan

Verifikasi ini memuat langsung model dari `gwo_model.joblib` yang tersimpan pada aplikasi. Optimasi GWO tidak dijalankan ulang dan tidak ada model baru yang dibuat.

Data testing dibentuk dengan pipeline utama penelitian: nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diubah menjadi NaN; train-test split 80:20 dengan `random_state=42`; imputasi median di-fit pada data training; dan StandardScaler di-fit pada data training.

## Parameter Model dari gwo_model.joblib

| Parameter | Nilai |
| --- | --- |
| Sumber model | gwo_model.joblib |
| Class model | DecisionTreeClassifier |
| max_depth | 4 |
| min_samples_leaf | 22 |
| criterion | gini |
| random_state | 42 |
| min_samples_split | 2 |
| splitter | best |

## Confusion Matrix

| Komponen | Jumlah | Keterangan |
| --- | --- | --- |
| TN | 66 | Aktual Tidak Diabetes, prediksi Tidak Diabetes |
| FP | 33 | Aktual Tidak Diabetes, prediksi Diabetes |
| FN | 13 | Aktual Diabetes, prediksi Tidak Diabetes |
| TP | 42 | Aktual Diabetes, prediksi Diabetes |

## Metrik Evaluasi

| Metrik | Hasil Program | Hasil Manual | Sama |
| --- | --- | --- | --- |
| Accuracy | 0.701299 | 0.701299 | Ya |
| Precision | 0.560000 | 0.560000 | Ya |
| Recall | 0.763636 | 0.763636 | Ya |
| F1-Score | 0.646154 | 0.646154 | Ya |

## 10 Data Testing Pertama

| No | Index Dataset | Outcome Aktual | Prediksi GWO | Status |
| --- | --- | --- | --- | --- |
| 1 | 668 | Tidak Diabetes | Diabetes | Salah |
| 2 | 324 | Tidak Diabetes | Tidak Diabetes | Benar |
| 3 | 624 | Tidak Diabetes | Tidak Diabetes | Benar |
| 4 | 690 | Tidak Diabetes | Tidak Diabetes | Benar |
| 5 | 473 | Tidak Diabetes | Tidak Diabetes | Benar |
| 6 | 204 | Tidak Diabetes | Diabetes | Salah |
| 7 | 97 | Tidak Diabetes | Tidak Diabetes | Benar |
| 8 | 336 | Tidak Diabetes | Diabetes | Salah |
| 9 | 568 | Tidak Diabetes | Diabetes | Salah |
| 10 | 148 | Tidak Diabetes | Diabetes | Salah |

## Perhitungan Manual

| Metrik | Rumus | Substitusi | Perhitungan | Hasil |
| --- | --- | --- | --- | --- |
| Accuracy | Accuracy = (TP + TN) / (TP + TN + FP + FN) | (42 + 66) / (42 + 66 + 33 + 13) | 108 / 154 | 0.701299 |
| Precision | Precision = TP / (TP + FP) | 42 / (42 + 33) | 42 / 75 | 0.560000 |
| Recall | Recall = TP / (TP + FN) | 42 / (42 + 13) | 42 / 55 | 0.763636 |
| F1-Score | F1 = 2 x (Precision x Recall) / (Precision + Recall) | 2 x (0.560000 x 0.763636) / (0.560000 + 0.763636) | 0.855273 / 1.323636 | 0.646154 |

### Langkah Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy = (42 + 66) / (42 + 66 + 33 + 13)
Accuracy = 108 / 154
Accuracy = 0.701299
```

### Langkah Precision

```text
Precision = TP / (TP + FP)
Precision = 42 / (42 + 33)
Precision = 42 / 75
Precision = 0.560000
```

### Langkah Recall

```text
Recall = TP / (TP + FN)
Recall = 42 / (42 + 13)
Recall = 42 / 55
Recall = 0.763636
```

### Langkah F1-Score

```text
F1 = 2 x (Precision x Recall) / (Precision + Recall)
F1 = 2 x (0.560000 x 0.763636) / (0.560000 + 0.763636)
F1 = 0.855273 / 1.323636
F1 = 0.646154
```

## Kesimpulan

- Model yang dievaluasi berasal dari file `gwo_model.joblib` yang tersimpan pada aplikasi.
- Optimasi GWO tidak dijalankan ulang dan model baru tidak dibuat.
- Perhitungan manual Accuracy, Precision, Recall, dan F1-Score sama dengan hasil program.