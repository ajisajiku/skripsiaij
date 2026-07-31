# Laporan Eksperimen Bab 4

Catatan project: `app.py` dan `model_utils.py` menggunakan `StandardScaler`. Script eksperimen Bab 4 ini sudah disesuaikan agar mengikuti pipeline utama aplikasi.


## Struktur Project

- `app.py`: file utama aplikasi Streamlit dan navigasi modul.
- `model_utils.py`: preprocessing, training baseline, RandomizedSearchCV, dan evaluasi UI.
- `gwo_optimizer.py`: fungsi fitness dan generator proses Grey Wolf Optimizer.
- `comparison.py`: script pembanding metrik baseline, GWO, dan RandomizedSearchCV.
- `diabetes_prediction.py`: script training GWO dan pembuatan SHAP summary plot.
- `interactive_prediction.py`: modul UI prediksi interaktif Streamlit.
- `diabetes.csv`: dataset Pima Indians Diabetes.
- `bab4_experiment_report.py`: script eksperimen Bab 4 terpisah.


## Alur Program

1. Dataset dimuat dari `diabetes.csv`.
2. Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diperlakukan sebagai missing value, lalu diganti menjadi NaN.
3. Fitur dipisahkan dari target `Outcome`.
4. Data dibagi menjadi training dan testing dengan rasio 80:20 menggunakan `random_state=42`.
5. Missing value diimputasi memakai median yang dipelajari dari data training.
6. Fitur distandardisasi memakai `StandardScaler` yang di-fit pada data training.
7. Decision Tree baseline dilatih menggunakan parameter default scikit-learn dengan `random_state=42`.
8. RandomizedSearchCV mencari kombinasi `criterion`, `max_depth`, dan `min_samples_leaf` dengan scoring recall.
9. GWO mencari `max_depth` dan `min_samples_leaf` berdasarkan fitness `1 - mean recall 5-fold CV`.
10. Model dievaluasi pada data testing menggunakan confusion matrix, accuracy, precision, recall, dan F1-score.


## 10 Data Awal Sebelum Preprocessing

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 148 | 72 | 35 | 0 | 33.6000 | 0.6270 | 50 | 1 |
| 1 | 85 | 66 | 29 | 0 | 26.6000 | 0.3510 | 31 | 0 |
| 8 | 183 | 64 | 0 | 0 | 23.3000 | 0.6720 | 32 | 1 |
| 1 | 89 | 66 | 23 | 94 | 28.1000 | 0.1670 | 21 | 0 |
| 0 | 137 | 40 | 35 | 168 | 43.1000 | 2.2880 | 33 | 1 |
| 5 | 116 | 74 | 0 | 0 | 25.6000 | 0.2010 | 30 | 0 |
| 3 | 78 | 50 | 32 | 88 | 31.0000 | 0.2480 | 26 | 1 |
| 10 | 115 | 0 | 0 | 0 | 35.3000 | 0.1340 | 29 | 0 |
| 2 | 197 | 70 | 45 | 543 | 30.5000 | 0.1580 | 53 | 1 |
| 8 | 125 | 96 | 0 | 0 | 0.0000 | 0.2320 | 54 | 1 |


## Missing Value Setelah Nilai 0 Dianggap Missing

| Atribut | Jumlah Missing Value |
| --- | --- |
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |


## Median Tiap Atribut

| Atribut | Median Data Training |
| --- | --- |
| Glucose | 118.0000 |
| BloodPressure | 72.0000 |
| SkinThickness | 28.5000 |
| Insulin | 120.0000 |
| BMI | 32.0000 |


## 10 Data Training Setelah Imputasi Median

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0000 | 84.0000 | 72.0000 | 28.5000 | 120.0000 | 32.0000 | 0.3040 | 21.0000 |
| 9.0000 | 112.0000 | 82.0000 | 24.0000 | 120.0000 | 28.2000 | 1.2820 | 50.0000 |
| 1.0000 | 139.0000 | 46.0000 | 19.0000 | 83.0000 | 28.7000 | 0.6540 | 22.0000 |
| 0.0000 | 161.0000 | 50.0000 | 28.5000 | 120.0000 | 21.9000 | 0.2540 | 65.0000 |
| 6.0000 | 134.0000 | 80.0000 | 37.0000 | 370.0000 | 46.2000 | 0.2380 | 46.0000 |
| 1.0000 | 130.0000 | 70.0000 | 13.0000 | 105.0000 | 25.9000 | 0.4720 | 22.0000 |
| 4.0000 | 132.0000 | 72.0000 | 28.5000 | 120.0000 | 32.9000 | 0.3020 | 23.0000 |
| 10.0000 | 161.0000 | 68.0000 | 23.0000 | 132.0000 | 25.5000 | 0.3260 | 47.0000 |
| 1.0000 | 108.0000 | 60.0000 | 46.0000 | 178.0000 | 35.5000 | 0.4150 | 24.0000 |
| 1.0000 | 80.0000 | 55.0000 | 28.5000 | 120.0000 | 19.1000 | 0.2580 | 21.0000 |


## 10 Data Training Setelah StandardScaler

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.5264 | -1.2572 | -0.0190 | -0.0081 | -0.2045 | -0.0502 | -0.4907 | -1.0359 |
| 1.5880 | -0.3263 | 0.8082 | -0.5437 | -0.2045 | -0.5986 | 2.4150 | 1.4871 |
| -0.8285 | 0.5713 | -2.1696 | -1.1387 | -0.6222 | -0.5264 | 0.5492 | -0.9489 |
| -1.1305 | 1.3027 | -1.8388 | -0.0081 | -0.2045 | -1.5077 | -0.6393 | 2.7921 |
| 0.6819 | 0.4051 | 0.6427 | 1.0034 | 2.6179 | 1.9988 | -0.6868 | 1.1391 |
| -0.8285 | 0.2721 | -0.1844 | -1.8527 | -0.3739 | -0.9305 | 0.0084 | -0.9489 |
| 0.0777 | 0.3386 | -0.0190 | -0.0081 | -0.2045 | 0.0796 | -0.4967 | -0.8619 |
| 1.8901 | 1.3027 | -0.3499 | -0.6627 | -0.0690 | -0.9882 | -0.4254 | 1.2261 |
| -0.8285 | -0.4593 | -1.0116 | 2.0744 | 0.4503 | 0.4548 | -0.1609 | -0.7749 |
| -0.8285 | -1.3902 | -1.4252 | -0.0081 | -0.2045 | -1.9117 | -0.6274 | -1.0359 |


## Rumus StandardScaler

| Metode | Rumus | Keterangan |
| --- | --- | --- |
| StandardScaler | z = (x - mean) / standard deviation | mean dan standard deviation dihitung dari data training |


## Contoh Perhitungan Manual StandardScaler

| Atribut | Nilai x | Mean Training | Standard Deviation Training | Perhitungan | Hasil z | Nilai dari StandardScaler |
| --- | --- | --- | --- | --- | --- | --- |
| Glucose | 84.0000 | 121.8160 | 30.0795 | (84.0000 - 121.8160) / 30.0795 | -1.2572 | -1.2572 |


## Jumlah Data Training dan Testing

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


## 10 Contoh Data Testing dan Prediksi

| Nomor | Nilai Aktual | Prediksi Baseline | Prediksi RandomizedSearch | Prediksi GWO |
| --- | --- | --- | --- | --- |
| 1 | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes | Diabetes |
| 2 | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes |
| 3 | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes |
| 4 | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes |
| 5 | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes |
| 6 | Tidak Diabetes | Diabetes | Diabetes | Diabetes |
| 7 | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes | Tidak Diabetes |
| 8 | Tidak Diabetes | Diabetes | Diabetes | Diabetes |
| 9 | Tidak Diabetes | Diabetes | Diabetes | Diabetes |
| 10 | Tidak Diabetes | Diabetes | Diabetes | Diabetes |


## Confusion Matrix

| Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Baseline DT | 76 | 23 | 19 | 36 |
| RandomizedSearch DT | 75 | 24 | 17 | 38 |
| GWO DT | 66 | 33 | 13 | 42 |


## Metrik Evaluasi

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Baseline DT | 0.7273 | 0.6102 | 0.6545 | 0.6316 |
| RandomizedSearch DT | 0.7338 | 0.6129 | 0.6909 | 0.6496 |
| GWO DT | 0.7013 | 0.5600 | 0.7636 | 0.6462 |


## Perhitungan Manual Metrik

| Model | Accuracy Manual | Precision Manual | Recall Manual | F1-Score Manual |
| --- | --- | --- | --- | --- |
| Baseline DT | (36 + 76) / (36 + 76 + 23 + 19) = 0.7273 | 36 / (36 + 23) = 0.6102 | 36 / (36 + 19) = 0.6545 | 2 * (0.6102 * 0.6545) / (0.6102 + 0.6545) = 0.6316 |
| RandomizedSearch DT | (38 + 75) / (38 + 75 + 24 + 17) = 0.7338 | 38 / (38 + 24) = 0.6129 | 38 / (38 + 17) = 0.6909 | 2 * (0.6129 * 0.6909) / (0.6129 + 0.6909) = 0.6496 |
| GWO DT | (42 + 66) / (42 + 66 + 33 + 13) = 0.7013 | 42 / (42 + 33) = 0.5600 | 42 / (42 + 13) = 0.7636 | 2 * (0.5600 * 0.7636) / (0.5600 + 0.7636) = 0.6462 |


## Parameter Terbaik RandomizedSearch

| Parameter | Nilai |
| --- | --- |
| min_samples_leaf | 30 |
| max_depth | 4 |
| criterion | entropy |


## 10 Contoh Iterasi RandomizedSearch

| Kombinasi Parameter | Mean Recall CV | Std Recall CV | Rank |
| --- | --- | --- | --- |
| {'min_samples_leaf': 11, 'max_depth': 18, 'criterion': 'entropy'} | 0.6338 | 0.0503 | 23 |
| {'min_samples_leaf': 4, 'max_depth': 8, 'criterion': 'gini'} | 0.5585 | 0.0753 | 66 |
| {'min_samples_leaf': 34, 'max_depth': 7, 'criterion': 'entropy'} | 0.6298 | 0.0803 | 24 |
| {'min_samples_leaf': 6, 'max_depth': 19, 'criterion': 'gini'} | 0.5445 | 0.0603 | 71 |
| {'min_samples_leaf': 40, 'max_depth': 6, 'criterion': 'entropy'} | 0.6670 | 0.0973 | 5 |
| {'min_samples_leaf': 24, 'max_depth': 6, 'criterion': 'entropy'} | 0.5777 | 0.0470 | 56 |
| {'min_samples_leaf': 39, 'max_depth': 19, 'criterion': 'gini'} | 0.6197 | 0.0680 | 32 |
| {'min_samples_leaf': 32, 'max_depth': 15, 'criterion': 'entropy'} | 0.6393 | 0.0966 | 17 |
| {'min_samples_leaf': 16, 'max_depth': 2, 'criterion': 'gini'} | 0.5066 | 0.1587 | 89 |
| {'min_samples_leaf': 24, 'max_depth': 7, 'criterion': 'entropy'} | 0.5734 | 0.0628 | 57 |


## Parameter GWO

| Parameter | Nilai |
| --- | --- |
| Jumlah wolf | 10 |
| Jumlah iterasi | 50 |
| Fungsi fitness | 1 - mean recall 5-fold CV |
| Dimensi | 2 |
| Lower bound | [1, 1] |
| Upper bound | [20, 50] |


## Perkembangan Fitness GWO

| Iterasi | Best Fitness (1 - Recall) | Best Recall CV | Max Depth | Min Samples Leaf |
| --- | --- | --- | --- | --- |
| 1 | 0.3090 | 0.6910 | 4 | 9 |
| 10 | 0.2538 | 0.7462 | 4 | 22 |
| 20 | 0.2538 | 0.7462 | 4 | 22 |
| 30 | 0.2538 | 0.7462 | 4 | 22 |
| 40 | 0.2538 | 0.7462 | 4 | 22 |
| 50 | 0.2538 | 0.7462 | 4 | 22 |


## Parameter Terbaik GWO

| Parameter | Nilai |
| --- | --- |
| criterion | gini |
| max_depth | 4 |
| min_samples_leaf | 22 |
| random_state | 42 |


## Narasi Singkat Akademik

Data awal menunjukkan beberapa atribut klinis memiliki nilai 0 yang secara medis tidak wajar, sehingga nilai tersebut diperlakukan sebagai missing value.
Imputasi median digunakan karena lebih stabil terhadap nilai ekstrem dibandingkan rata-rata.
StandardScaler mengubah fitur menjadi nilai z-score dengan rumus z = (x - mean) / standard deviation, sehingga fitur berada pada skala yang lebih sebanding sesuai pipeline utama aplikasi.
Pembagian data 80:20 menghasilkan data training untuk pembentukan model dan data testing untuk evaluasi generalisasi.
Baseline Decision Tree digunakan sebagai pembanding awal tanpa optimasi hyperparameter.
RandomizedSearchCV dan GWO sama-sama mengoptimasi hyperparameter dengan fokus pada recall agar model lebih sensitif dalam mengenali kelas diabetes.
Evaluasi akhir menggunakan confusion matrix serta accuracy, precision, recall, dan F1-score untuk membandingkan performa model secara menyeluruh.
