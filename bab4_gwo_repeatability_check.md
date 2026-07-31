# GWO Repeatability Check Bab 4

Verifikasi ini menjalankan proses GWO sebanyak 10 kali dengan pipeline utama aplikasi: train-test split 80:20 `random_state=42`, imputasi median dari data training, `StandardScaler` dari data training, lalu `run_gwo(calculate_fitness, ...)` dari `gwo_optimizer.py`.

Parameter GWO mengikuti aplikasi:

- jumlah wolf = 10
- jumlah iterasi = 20
- dimensi = 2
- lower bound = [1, 1]
- upper bound = [20, 50]
- fitness = 1 - mean recall 5-fold CV
- parameter Decision Tree yang dioptimasi = max_depth dan min_samples_leaf

Catatan: aplikasi tidak menetapkan seed eksplisit untuk GWO. Karena posisi awal wolf memakai `np.random.uniform` dan update posisi memakai `random.random`, hasil dapat berubah antar run.

## Hasil 10 Kali Run GWO

| Run | max_depth | min_samples_leaf | Best Fitness | Best Recall CV | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 21 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 2 | 4 | 22 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 3 | 4 | 22 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 4 | 4 | 21 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 5 | 4 | 21 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 6 | 4 | 22 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 7 | 6 | 11 | 0.291362 | 0.708638 | 0.740260 | 0.631579 | 0.654545 | 0.642857 |
| 8 | 4 | 21 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |
| 9 | 8 | 32 | 0.342193 | 0.657807 | 0.772727 | 0.692308 | 0.654545 | 0.672897 |
| 10 | 4 | 22 | 0.253821 | 0.746179 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |

## Ringkasan Recall

| Statistik | Nilai |
| --- | --- |
| Recall terbaik | 0.763636 |
| Recall terburuk | 0.654545 |
| Recall rata-rata | 0.741818 |
| Jumlah run unik berdasarkan parameter | 4 |

## Model GWO Tersimpan

| Keterangan | Status | max_depth | min_samples_leaf | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gwo_model.joblib | Berhasil dimuat | 4 | 22 | 0.701299 | 0.560000 | 0.763636 | 0.646154 |

## Kesimpulan

- Hasil GWO bersifat stochastic: Ya.
- Recall dapat berubah antar run: Ya.
- Recall terbaik dari 10 run adalah 0.763636.
- Recall terburuk dari 10 run adalah 0.654545.
- Recall rata-rata dari 10 run adalah 0.741818.
- Untuk Bab 4, hasil GWO sebaiknya dilaporkan dengan catatan bahwa tanpa seed eksplisit, hasil optimasi dapat berubah ketika proses dijalankan ulang.