# Verifikasi RandomizedSearchCV Bab 4

Pipeline yang digunakan sama dengan aplikasi utama: train-test split 80:20 dengan `random_state=42`, nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diubah menjadi NaN, imputasi median berdasarkan data training, StandardScaler di-fit pada data training, lalu RandomizedSearchCV dilatih pada data training hasil preprocessing.

Fungsi aplikasi yang menjadi acuan adalah `train_randomized_search_model(X_train_scaled, y_train)` pada `model_utils.py`.

```python
param_dist = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(1, 21)), 'min_samples_leaf': list(range(1, 51))}
rand_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_dist, n_iter=100, cv=5, scoring='recall', random_state=42, n_jobs=-1)
```

Catatan: `n_jobs` pada aplikasi adalah `-1` untuk paralelisasi. Pada script verifikasi ini `n_jobs=1` digunakan agar eksekusi non-UI tidak bergantung pada multiprocessing Windows; search space, `n_iter`, `cv`, `scoring`, dan `random_state` tetap sama sehingga kombinasi dan hasil model tetap konsisten.

## Ruang Pencarian Parameter

| Parameter | Nilai/Rentang | Jumlah Nilai | Status |
| --- | --- | --- | --- |
| criterion | ['gini', 'entropy'] | 2 | Dicari RandomizedSearchCV |
| max_depth | 1 sampai 20 | 20 | Dicari RandomizedSearchCV |
| min_samples_leaf | 1 sampai 50 | 50 | Dicari RandomizedSearchCV |
| min_samples_split | 2 | 1 | Tidak dicari; memakai default DecisionTreeClassifier |

## Konfigurasi RandomizedSearchCV

| Konfigurasi | Nilai |
| --- | --- |
| Jumlah kombinasi mungkin | 2000 |
| n_iter | 100 |
| cv | 5 |
| scoring | recall |
| random_state | 42 |
| n_jobs pada aplikasi utama | -1 |
| n_jobs pada script verifikasi | 1 |

## 10 Kombinasi Hyperparameter Pertama yang Dicoba

| No | criterion | max_depth | min_samples_split | min_samples_leaf | Mean Recall CV | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | entropy | 18 | 2 | 11 | 0.633776 | 23 |
| 2 | gini | 8 | 2 | 4 | 0.558472 | 66 |
| 3 | entropy | 7 | 2 | 34 | 0.629790 | 24 |
| 4 | gini | 19 | 2 | 6 | 0.544518 | 71 |
| 5 | entropy | 6 | 2 | 40 | 0.666999 | 5 |
| 6 | entropy | 6 | 2 | 24 | 0.577741 | 56 |
| 7 | gini | 19 | 2 | 39 | 0.619712 | 32 |
| 8 | entropy | 15 | 2 | 32 | 0.639313 | 17 |
| 9 | gini | 2 | 2 | 16 | 0.506645 | 89 |
| 10 | entropy | 7 | 2 | 24 | 0.573422 | 57 |

## Contoh Pemilihan Kombinasi Hyperparameter

Kombinasi ke-1 yang dicoba oleh RandomizedSearchCV adalah:

- criterion = `entropy`
- max_depth = `18`
- min_samples_split = `2` (default DecisionTreeClassifier, tidak termasuk search space)
- min_samples_leaf = `11`
- mean recall CV = `0.633776`

RandomizedSearchCV mengevaluasi kombinasi tersebut menggunakan 5-fold cross-validation pada data training dengan scoring `recall`. Nilai mean recall CV dari setiap kombinasi dibandingkan, lalu kombinasi dengan recall CV tertinggi dipilih sebagai `best_params_`.

## Hyperparameter Terbaik

| Parameter | Nilai |
| --- | --- |
| criterion | entropy |
| max_depth | 4 |
| min_samples_split | 2 |
| min_samples_leaf | 30 |
| best_score_ recall CV | 0.723367 |

## Confusion Matrix RandomizedSearch

| Komponen | Jumlah | Keterangan |
| --- | --- | --- |
| TN | 75 | Aktual Tidak Diabetes, prediksi Tidak Diabetes |
| FP | 24 | Aktual Tidak Diabetes, prediksi Diabetes |
| FN | 17 | Aktual Diabetes, prediksi Tidak Diabetes |
| TP | 38 | Aktual Diabetes, prediksi Diabetes |

## Metrik Evaluasi Program dan Manual

| Metrik | Hasil Program | Hasil Manual | Sama |
| --- | --- | --- | --- |
| Accuracy | 0.733766 | 0.733766 | Ya |
| Precision | 0.612903 | 0.612903 | Ya |
| Recall | 0.690909 | 0.690909 | Ya |
| F1-Score | 0.649573 | 0.649573 | Ya |

## 10 Data Testing Pertama dan Prediksi RandomizedSearch

| No | Index Dataset | Outcome Aktual | Prediksi RandomizedSearch | Status |
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

## Perhitungan Manual Metrik

| Metrik | Rumus | Substitusi | Perhitungan | Hasil |
| --- | --- | --- | --- | --- |
| Accuracy | Accuracy = (TP + TN) / (TP + TN + FP + FN) | (38 + 75) / (38 + 75 + 24 + 17) | 113 / 154 | 0.733766 |
| Precision | Precision = TP / (TP + FP) | 38 / (38 + 24) | 38 / 62 | 0.612903 |
| Recall | Recall = TP / (TP + FN) | 38 / (38 + 17) | 38 / 55 | 0.690909 |
| F1-Score | F1 = 2 x (Precision x Recall) / (Precision + Recall) | 2 x (0.612903 x 0.690909) / (0.612903 + 0.690909) | 0.846921 / 1.303812 | 0.649573 |

### Langkah Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy = (38 + 75) / (38 + 75 + 24 + 17)
Accuracy = 113 / 154
Accuracy = 0.733766
```

### Langkah Precision

```text
Precision = TP / (TP + FP)
Precision = 38 / (38 + 24)
Precision = 38 / 62
Precision = 0.612903
```

### Langkah Recall

```text
Recall = TP / (TP + FN)
Recall = 38 / (38 + 17)
Recall = 38 / 55
Recall = 0.690909
```

### Langkah F1-Score

```text
F1 = 2 x (Precision x Recall) / (Precision + Recall)
F1 = 2 x (0.612903 x 0.690909) / (0.612903 + 0.690909)
F1 = 0.846921 / 1.303812
F1 = 0.649573
```

## Perbandingan dengan Baseline

| Metrik | Baseline | RandomizedSearch | Perubahan |
| --- | --- | --- | --- |
| Recall | 0.654545 | 0.690909 | 0.036364 |
| Accuracy | 0.727273 | 0.733766 | 0.006494 |

## Kesimpulan

- Parameter terbaik yang ditemukan adalah `criterion=entropy`, `max_depth=4`, `min_samples_leaf=30`, dengan `min_samples_split=2` sebagai nilai default.
- Best score selama pencarian adalah recall CV `0.723367`.
- Recall meningkat dari baseline `0.654545` menjadi `0.690909`, atau naik `0.036364`.
- Accuracy meningkat dari baseline `0.727273` menjadi `0.733766`, atau naik `0.006494`.
- RandomizedSearch berhasil meningkatkan kemampuan deteksi diabetes dibanding baseline berdasarkan kenaikan recall, meskipun peningkatannya relatif moderat.