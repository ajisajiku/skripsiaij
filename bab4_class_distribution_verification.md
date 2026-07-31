# Verifikasi Distribusi Kelas Outcome Bab 4

Pipeline split yang digunakan mengikuti aplikasi utama, yaitu `train_test_split(X, y, test_size=0.2, random_state=42)`. Parameter `stratify` tidak digunakan pada pipeline utama.

Jumlah data total dataset: 768

## Distribusi Kelas Outcome

| Subset | Tidak Diabetes | Diabetes | Total | Persentase Tidak Diabetes | Persentase Diabetes |
| --- | --- | --- | --- | --- | --- |
| Dataset Penuh | 500 | 268 | 768 | 65.10% | 34.90% |
| Training | 401 | 213 | 614 | 65.31% | 34.69% |
| Testing | 99 | 55 | 154 | 64.29% | 35.71% |

## Parameter Train-Test Split

| Parameter | Nilai |
| --- | --- |
| test_size | 0.2 |
| random_state | 42 |
| stratify | Tidak digunakan / None |
| Kode aktual | train_test_split(X, y, test_size=0.2, random_state=42) |

## Selisih Distribusi Kelas

| Perbandingan | Selisih Tidak Diabetes | Selisih Diabetes | Selisih Absolut Diabetes |
| --- | --- | --- | --- |
| Training - Dataset Penuh | 0.21% | -0.21% | 0.21% |
| Testing - Dataset Penuh | -0.82% | 0.82% | 0.82% |
| Testing - Training | -1.02% | 1.02% | 1.02% |

## Interpretasi

Karena `stratify` tidak digunakan, pembagian data training dan testing dilakukan secara acak berdasarkan `random_state=42`, bukan dengan pemaksaan proporsi kelas yang sama. Distribusi kelas tetap relatif serupa: proporsi Diabetes pada dataset penuh adalah 34.90%, pada data training 34.69%, dan pada data testing 35.71%. Selisih terbesar proporsi Diabetes antara subset yang dibandingkan adalah 1.02%.

## Narasi Akademik Singkat

Dataset Pima Indians Diabetes yang digunakan berjumlah 768 data, terdiri dari 500 data kelas Tidak Diabetes dan 268 data kelas Diabetes. Setelah proses pembagian data menggunakan rasio 80:20 dengan `random_state=42`, data training berjumlah 614 data dan data testing berjumlah 154 data. Pada data training terdapat 401 data Tidak Diabetes dan 213 data Diabetes, sedangkan pada data testing terdapat 99 data Tidak Diabetes dan 55 data Diabetes. Parameter `stratify` tidak digunakan dalam proses pembagian data, sehingga proporsi kelas tidak dipaksa sama persis dengan dataset penuh. Namun, distribusi kelas pada training dan testing masih relatif serupa dengan dataset penuh, sehingga pembagian data tetap dapat digunakan untuk proses evaluasi model.

## Kesimpulan

- Jumlah data total dataset adalah 768.
- Train-test split menghasilkan 614 data training dan 154 data testing.
- Parameter `stratify` tidak digunakan.
- Distribusi kelas training dan testing masih relatif serupa terhadap dataset penuh berdasarkan selisih persentase kelas Diabetes.