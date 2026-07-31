# Penjelasan Grey Wolf Optimizer (GWO)

Berikut adalah penjelasan mengenai cara kerja Grey Wolf Optimizer (GWO), sebuah algoritma optimasi metaheuristik.

## 1. Apa itu Grey Wolf Optimizer (GWO)?

*   GWO adalah algoritma optimasi metaheuristik yang terinspirasi dari alam.
*   Algoritma ini meniru mekanisme berburu dan hierarki sosial serigala abu-abu di alam liar.
*   Tujuannya adalah untuk menemukan solusi optimal (dalam kasus kita, set *hyperparameter* terbaik) untuk suatu masalah.

## 2. Hierarki Sosial Serigala Abu-abu (Analogi Optimasi):

Serigala abu-abu memiliki hierarki sosial yang ketat, yang diadaptasi dalam algoritma GWO:

*   **Alpha (α):** Serigala pemimpin, yang membuat keputusan tentang perburuan. Dalam optimasi, ini adalah **solusi terbaik** yang ditemukan sejauh ini.
*   **Beta (β):** Serigala kedua dalam komando, membantu alpha dalam pengambilan keputusan. Dalam optimasi, ini adalah **solusi terbaik kedua**.
*   **Delta (δ):** Serigala bawahan, membantu alpha dan beta. Dalam optimasi, ini adalah **solusi terbaik ketiga**.
*   **Omega (ω):** Serigala dengan peringkat terendah, mengikuti serigala lain. Dalam optimasi, ini adalah **solusi kandidat** lainnya.

## 3. Bagaimana GWO Mengoptimasi *Hyperparameter* (Analogi Mekanisme Berburu):

*   **Inisialisasi:**
    *   Populasi 'serigala abu-abu' (solusi kandidat, di mana setiap serigala mewakili satu set *hyperparameter* untuk Decision Tree) diinisialisasi secara acak dalam ruang pencarian yang ditentukan (misalnya, rentang untuk `max_depth` dan `min_samples_leaf`).
*   **Mengepung Mangsa (Eksplorasi):**
    *   Serigala-serigala mencoba mengepung 'mangsa' (set *hyperparameter* optimal yang memaksimalkan recall). Mereka tidak tahu lokasi pasti mangsa, tetapi alpha, beta, dan delta memiliki perkiraan terbaik.
    *   Posisi serigala omega diperbarui berdasarkan posisi alpha, beta, dan delta. Ini mendorong eksplorasi di sekitar solusi terbaik yang ditemukan sejauh ini.
*   **Berburu (Eksploitasi):**
    *   Alpha, beta, dan delta memandu perburuan. Serigala omega memperbarui posisi mereka berdasarkan posisi rata-rata dari tiga serigala terbaik. Ini mendorong pencarian menuju wilayah yang menjanjikan di ruang pencarian.
*   **Menyerang Mangsa (Konvergensi):**
    *   Seiring berjalannya iterasi, parameter `a` (yang menurun secara linear dari 2 menjadi 0) mempengaruhi keseimbangan antara eksplorasi (mencari secara luas) dan eksploitasi (penyesuaian halus di sekitar solusi terbaik). `a` yang besar mendorong eksplorasi, sedangkan `a` yang kecil mendorong eksploitasi.
*   **Fungsi Kebugaran (Fitness Function):**
    *   Untuk setiap set *hyperparameter* (posisi setiap serigala), 'fungsi kebugaran' dievaluasi. Dalam kasus kita, fungsi kebugaran menghitung `1.0 - rata-rata recall cross-validation` pada data pelatihan. Tujuannya adalah untuk **meminimalkan** nilai kebugaran ini (yang berarti memaksimalkan recall).
*   **Memperbarui Hierarki:**
    *   Setelah setiap iterasi, kebugaran semua serigala dievaluasi, dan serigala alpha, beta, dan delta diperbarui berdasarkan solusi terbaik baru yang ditemukan.
*   **Kriteria Penghentian:**
    *   Proses ini berlanjut untuk sejumlah iterasi yang telah ditentukan (`max_iterations`).

## 4. Manfaat GWO:

*   **Metaheuristik:** Mampu menjelajahi ruang pencarian yang kompleks dan non-linear secara efektif.
*   **Pencarian Global:** Cukup baik dalam menghindari *local optima* karena mekanisme eksplorasinya.
*   **Terinspirasi Alam:** Seringkali kuat dan efektif untuk berbagai masalah optimasi.

## 5. Peran dalam Proyek Ini:

*   Dalam proyek ini, GWO digunakan sebagai teknik optimasi lanjutan untuk menemukan `max_depth` dan `min_samples_leaf` terbaik untuk model Decision Tree. Tujuannya adalah untuk mencapai recall dan kemampuan generalisasi yang lebih baik lagi dibandingkan dengan `RandomizedSearchCV` atau model baseline, dengan memanfaatkan strategi pencarian yang terinspirasi dari perilaku serigala abu-abu.
