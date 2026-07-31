# Verifikasi Grey Wolf Optimizer (GWO) Bab 4

Verifikasi ini menggunakan pipeline utama aplikasi: train-test split 80:20 dengan `random_state=42`, imputasi median berdasarkan data training, `StandardScaler` berdasarkan data training, lalu optimasi GWO menggunakan fungsi pada project.

## File dan Fungsi GWO yang Digunakan

- `gwo_optimizer.py::calculate_fitness`: menghitung fitness kandidat parameter Decision Tree.
- `gwo_optimizer.py::run_gwo`: menjalankan proses Grey Wolf Optimizer.
- `app.py`: memanggil `run_gwo(calculate_fitness, ...)`, mengambil `alpha_params`, lalu melatih `DecisionTreeClassifier` final.

Potongan kode aktual:

```python
LOWER_BOUND, UPPER_BOUND, DIMENSIONS, NUM_WOLVES, MAX_ITERATIONS = [1, 1], [20, 50], 2, 10, 20
for update in run_gwo(calculate_fitness, LOWER_BOUND, UPPER_BOUND, DIMENSIONS, NUM_WOLVES, MAX_ITERATIONS, X_train_scaled, y_train):
    gwo_history.append(update['alpha_score'])
best_params = last_gwo_update['alpha_params']
gwo_depth = int(best_params[0])
gwo_leaf = int(best_params[1])
gwo_model = DecisionTreeClassifier(max_depth=gwo_depth, min_samples_leaf=gwo_leaf, criterion='gini', random_state=42)
gwo_model.fit(X_train_scaled, y_train)
```

Catatan penting: aplikasi aktual memakai `MAX_ITERATIONS=20`, sehingga iterasi 30, 40, dan 50 tidak dijalankan oleh menu GWO aplikasi. Kode GWO juga tidak menetapkan seed eksplisit untuk posisi wolf dan bilangan acak, sehingga hasil optimasi dapat berbeda antar eksekusi.

## Parameter GWO

| Parameter | Nilai |
| --- | --- |
| File fungsi fitness | gwo_optimizer.py::calculate_fitness |
| File fungsi optimizer | gwo_optimizer.py::run_gwo |
| Training model GWO | app.py membuat DecisionTreeClassifier(max_depth=gwo_depth, min_samples_leaf=gwo_leaf, criterion="gini", random_state=42) |
| Jumlah wolf / population size | 10 |
| Jumlah iterasi aktual aplikasi | 20 |
| Dimensi | 2 |
| Lower bound | [1, 1] |
| Upper bound | [20, 50] |
| Fungsi fitness | 1 - mean recall 5-fold cross-validation |
| Parameter DT yang dioptimasi | max_depth dan min_samples_leaf |
| Criterion pada fitness dan model final | gini |
| random_state Decision Tree | 42 |
| Seed GWO eksplisit pada aplikasi | Tidak ada |

## Rumus Fitness

Fungsi fitness pada `calculate_fitness` adalah:

```text
fitness = 1 - mean recall 5-fold cross-validation
```

Semakin kecil fitness, semakin baik kandidat parameter karena recall CV semakin tinggi.

## Contoh Perhitungan Fitness

| Komponen | Nilai |
| --- | --- |
| Best recall CV | 0.708638 |
| Rumus fitness | fitness = 1 - recall CV |
| Substitusi | fitness = 1 - 0.708638 |
| Fitness | 0.291362 |

## Perkembangan Fitness GWO

| Iterasi | Best Fitness | Best Recall CV | max_depth | min_samples_leaf | Keterangan |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.370875 | 0.629125 | 10 | 15 | Aktual dari run_gwo aplikasi |
| 10 | 0.309856 | 0.690144 | 6 | 13 | Aktual dari run_gwo aplikasi |
| 20 | 0.291362 | 0.708638 | 6 | 11 | Aktual dari run_gwo aplikasi |
| 30 | Tidak dijalankan | Tidak dijalankan | Tidak dijalankan | Tidak dijalankan | Aplikasi memakai MAX_ITERATIONS=20 |
| 40 | Tidak dijalankan | Tidak dijalankan | Tidak dijalankan | Tidak dijalankan | Aplikasi memakai MAX_ITERATIONS=20 |
| 50 | Tidak dijalankan | Tidak dijalankan | Tidak dijalankan | Tidak dijalankan | Aplikasi memakai MAX_ITERATIONS=20 |

## Parameter Terbaik Hasil GWO

| Parameter | Nilai |
| --- | --- |
| criterion | gini |
| max_depth | 6 |
| min_samples_leaf | 11 |
| random_state | 42 |
| min_samples_split | 2 |
| Best fitness | 0.291362 |
| Best recall CV | 0.708638 |

## 10 Data Testing Pertama dan Prediksi GWO

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

## Confusion Matrix GWO

| Komponen | Jumlah | Keterangan |
| --- | --- | --- |
| TN | 78 | Aktual Tidak Diabetes, prediksi Tidak Diabetes |
| FP | 21 | Aktual Tidak Diabetes, prediksi Diabetes |
| FN | 19 | Aktual Diabetes, prediksi Tidak Diabetes |
| TP | 36 | Aktual Diabetes, prediksi Diabetes |

## Metrik Evaluasi GWO

| Metrik | Hasil Program | Hasil Manual | Sama |
| --- | --- | --- | --- |
| Accuracy | 0.740260 | 0.740260 | Ya |
| Precision | 0.631579 | 0.631579 | Ya |
| Recall | 0.654545 | 0.654545 | Ya |
| F1-Score | 0.642857 | 0.642857 | Ya |

## Perhitungan Manual Metrik

| Metrik | Rumus | Substitusi | Perhitungan | Hasil |
| --- | --- | --- | --- | --- |
| Accuracy | Accuracy = (TP + TN) / (TP + TN + FP + FN) | (36 + 78) / (36 + 78 + 21 + 19) | 114 / 154 | 0.740260 |
| Precision | Precision = TP / (TP + FP) | 36 / (36 + 21) | 36 / 57 | 0.631579 |
| Recall | Recall = TP / (TP + FN) | 36 / (36 + 19) | 36 / 55 | 0.654545 |
| F1-Score | F1 = 2 x (Precision x Recall) / (Precision + Recall) | 2 x (0.631579 x 0.654545) / (0.631579 + 0.654545) | 0.826794 / 1.286124 | 0.642857 |

### Langkah Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy = (36 + 78) / (36 + 78 + 21 + 19)
Accuracy = 114 / 154
Accuracy = 0.740260
```

### Langkah Precision

```text
Precision = TP / (TP + FP)
Precision = 36 / (36 + 21)
Precision = 36 / 57
Precision = 0.631579
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
F1 = 2 x (0.631579 x 0.654545) / (0.631579 + 0.654545)
F1 = 0.826794 / 1.286124
F1 = 0.642857
```

## Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score | False Negative |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.727273 | 0.610169 | 0.654545 | 0.631579 | 19 |
| RandomizedSearch | 0.733766 | 0.612903 | 0.690909 | 0.649573 | 17 |
| GWO | 0.740260 | 0.631579 | 0.654545 | 0.642857 | 19 |

## Kesimpulan

- GWO menghasilkan parameter terbaik `criterion=gini`, `max_depth=6`, `min_samples_leaf=11`, dan `random_state=42` pada run verifikasi ini.
- Recall GWO dibanding baseline berubah sebesar `0.000000`.
- Recall GWO dibanding RandomizedSearch berubah sebesar `-0.036364`.
- False negative GWO dibanding baseline berubah sebesar `0` data.
- False negative GWO dibanding RandomizedSearch berubah sebesar `2` data.
- Accuracy GWO dibanding baseline berubah sebesar `0.012987` dan dibanding RandomizedSearch berubah sebesar `0.006494`.
- Fokus penelitian adalah recall, sehingga model yang lebih banyak mendeteksi kelas Diabetes lebih diutamakan meskipun accuracy dapat naik atau turun.