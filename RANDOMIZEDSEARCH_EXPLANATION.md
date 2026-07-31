# Penjelasan RandomizedSearchCV

Berikut adalah penjelasan mengenai cara kerja RandomizedSearchCV, sebuah metode optimasi *hyperparameter*.

## 1. Tujuan *Hyperparameter* dan Optimasi *Hyperparameter*

*   **_Hyperparameter_** adalah pengaturan atau konfigurasi dari algoritma *machine learning* itu sendiri, bukan parameter yang dipelajari dari data (seperti bobot dalam jaringan saraf). Contoh _hyperparameter_ untuk Decision Tree adalah `max_depth` (kedalaman maksimum pohon) atau `min_samples_leaf` (jumlah sampel minimum di daun).
*   **Mengapa Butuh Optimasi?** Pemilihan _hyperparameter_ yang tepat sangat krusial dan dapat berdampak signifikan pada kinerja model. Mengoptimasi _hyperparameter_ membantu model mencapai kinerja terbaik pada tugas yang diberikan.

## 2. Apa itu `RandomizedSearchCV`?

*   `RandomizedSearchCV` adalah teknik yang efisien untuk mencari kombinasi *hyperparameter* terbaik dalam ruang pencarian yang telah ditentukan.
*   Ini adalah metode yang lebih cepat dan efisien dibandingkan dengan `GridSearchCV` (yang mencoba setiap kombinasi secara *exhaustive*) untuk ruang pencarian yang besar.

## 3. Bagaimana `RandomizedSearchCV` Bekerja?

*   **Mendefinisikan Ruang Pencarian (Distribusi):**
    *   Alih-alih mencari setiap kombinasi potensial (seperti `GridSearchCV`), kita mendefinisikan **distribusi** dari nilai-nilai yang mungkin untuk setiap _hyperparameter_. Misalnya, untuk `max_depth`, kita bisa memberikan rentang [1, 2, ..., 20]. Untuk `criterion`, kita bisa memberikan daftar pilihan ['gini', 'entropy'].
*   **Pengambilan Sampel Acak (Random Sampling):**
    *   `RandomizedSearchCV` kemudian **mengambil sampel secara acak** sejumlah kombinasi _hyperparameter_ (`n_iter`, misalnya 100 kombinasi) dari distribusi yang telah kita definisikan itu.
    *   Pendekatan acak ini lebih efisien karena ia menjelajahi area yang beragam dalam ruang pencarian, alih-alih mencoba setiap titik.
*   **Evaluasi dengan *Cross-Validation*:**
    *   Untuk setiap kombinasi _hyperparameter_ yang diambil secara acak, model dilatih dan dievaluasi menggunakan **_cross-validation_** pada data pelatihan.
    *   (Ulangi penjelasan _cross-validation_ singkat di sini: data pelatihan dibagi menjadi beberapa 'lipatan', model dilatih pada sebagian lipatan dan divalidasi pada lipatan lainnya; skor rata-rata diambil).
*   **Metrik Penilaian (`scoring`):**
    *   Metrik penilaian spesifik (dalam kasus kita, `'recall'`) digunakan untuk mengukur kinerja setiap kombinasi _hyperparameter_. Tujuannya adalah untuk menemukan kombinasi yang memaksimalkan skor ini.
*   **Menemukan Parameter Terbaik:**
    *   Setelah mencoba sejumlah `n_iter` kombinasi, `RandomizedSearchCV` mengidentifikasi set kombinasi _hyperparameter_ yang menghasilkan skor *cross-validation* rata-rata terbaik.

## 4. Manfaat `RandomizedSearchCV`:

*   **Efisiensi:** Lebih cepat dibandingkan `GridSearchCV` untuk ruang pencarian yang besar karena tidak perlu mencoba setiap kombinasi.
*   **Efektivitas:** Seringkali dapat menemukan set _hyperparameter_ yang baik lebih cepat daripada pencarian *exhaustive*.
*   **Eksplorasi:** Dapat menjelajahi rentang nilai yang lebih luas jika distribusi *hyperparameter* didefinisikan dengan baik.

## 5. Peran dalam Proyek Ini:

*   Dalam proyek ini, `RandomizedSearchCV` digunakan untuk secara sistematis menemukan *hyperparameter* yang lebih baik untuk model Decision Tree dibanding parameter *default*. Tujuannya adalah untuk meningkatkan skor recall dan kemampuan generalisasi model dibandingkan dengan model baseline, tanpa harus menguji setiap kemungkinan kombinasi secara berlebihan.

## Analogi:

*   "Jika `GridSearchCV` seperti mencoba setiap hidangan di menu restoran untuk menemukan hidangan favorit Anda, `RandomizedSearchCV` seperti memilih beberapa hidangan secara acak selama beberapa kali kunjungan, dan seringkali Anda bisa menemukan hidangan favorit baru Anda lebih cepat, terutama jika menunya sangat besar."
