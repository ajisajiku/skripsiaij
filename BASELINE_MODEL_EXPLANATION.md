# Penjelasan Model Baseline

Berikut adalah penjelasan mengenai alur kerja dan proses perhitungan model baseline, yang dapat digunakan saat presentasi.

## 1. Apa itu Model Baseline?

*   Dalam konteks ini, Model Baseline adalah versi **sederhana dan tidak dioptimasi** dari model Decision Tree. Ini adalah model pertama yang kami latih untuk mendapatkan gambaran dasar tentang kinerja yang bisa kami harapkan.
*   Tujuannya adalah sebagai **tolok ukur (benchmark)**. Kami akan membandingkan kinerja model-model yang telah kami optimasi (menggunakan RandomizedSearchCV dan GWO) terhadap model baseline ini untuk melihat seberapa besar peningkatan yang kami capai.

## 2. Persiapan Data (Sama untuk Semua Model):

*   Sebelum melatih model baseline, data kami melalui serangkaian langkah pra-pemrosesan yang sama untuk semua model:
    *   **Penanganan Nilai Hilang:** Kami mengganti nilai 0 yang tidak valid (misalnya, tekanan darah 0) dengan nilai yang hilang (`NaN`), lalu mengisi nilai-nilai hilang tersebut dengan median dari setiap kolom.
    *   **Penskalaan Fitur:** Kami menormalisasi fitur-fitur numerik menggunakan `StandardScaler` agar semua fitur memiliki skala yang serupa, yang penting untuk kinerja banyak algoritma *machine learning*.
    *   **Pembagian Data:** Data kemudian dibagi menjadi **data pelatihan (training set)** dan **data pengujian (testing set)**. Model baseline hanya dilatih pada data pelatihan.

## 3. Pilihan Model: Decision Tree:

*   Kami memilih Decision Tree sebagai model dasar karena sifatnya yang intuitif dan mudah diinterpretasikan. Ini adalah algoritma klasifikasi yang membuat keputusan berdasarkan serangkaian pertanyaan (cabang pohon).

## 4. Proses Pelatihan Model Baseline:

*   Model Decision Tree baseline ini dilatih langsung pada **data pelatihan yang sudah diproses**.
*   Yang paling penting, model ini menggunakan **parameter *default***. Artinya, kami tidak melakukan penyesuaian *hyperparameter* apa pun pada tahap ini. Pohon keputusan dibiarkan tumbuh secara alami, seringkali menjadi sangat dalam dan kompleks.
*   Kami menggunakan `random_state=42` untuk memastikan hasil pelatihan dapat direproduksi.

## 5. Evaluasi Awal (Cross-Validation pada Data Pelatihan):

*   Setelah model dilatih, kami melakukan **evaluasi awal menggunakan *cross-validation*** pada **data pelatihan**. Ini memberikan kami 'Rata-rata Recall (Cross-Validation)' (misalnya, 0.7000).
*   Ini adalah estimasi internal tentang seberapa baik model berkinerja pada data yang sudah pernah dilihatnya, dan berfungsi sebagai indikator awal sebelum kita melihat kinerja pada data yang benar-benar baru.

## 6. Karakteristik dan Potensi Masalah Model Baseline:

*   **Kompleksitas Tinggi:** Karena menggunakan parameter *default*, pohon keputusan cenderung sangat dalam dan memiliki banyak cabang.
*   **Risiko *Overfitting*:** Pohon yang terlalu kompleks berisiko mengalami *overfitting*. Ini berarti model mungkin terlalu 'menghafal' pola-pola spesifik di data pelatihan, termasuk *noise*, dan akibatnya, kinerjanya bisa buruk saat dihadapkan pada data baru yang belum pernah dilihat.
*   **Visualisasi:** Jika Anda melihat visualisasi pohon keputusan untuk model baseline, Anda akan melihat betapa rumitnya strukturnya.

## 7. Peran Model Baseline dalam Proyek Ini:

*   Model baseline ini adalah **titik referensi** kami. Kami berharap model-model yang dioptimasi (RandomizedSearchCV dan GWO) akan menunjukkan kinerja yang lebih baik, terutama dalam hal kemampuan generalisasi pada data baru, dibandingkan dengan model baseline ini. Ini akan menunjukkan nilai dari proses optimasi *hyperparameter* yang kami lakukan.

## 8. Bagaimana Model Decision Tree Membuat Prediksi (Proses 'Perhitungan' Internal):

*   Bayangkan Decision Tree sebagai **serangkaian pertanyaan 'Ya/Tidak'**, mirip seperti *flowchart* atau pohon keputusan yang kita buat secara manual.
*   Setiap kotak (disebut 'node') dalam pohon mewakili **pertanyaan tentang salah satu fitur pasien** (misalnya, 'Apakah kadar Glukosa pasien lebih dari 120 mg/dL?').
*   Berdasarkan jawaban dari pertanyaan tersebut (Ya atau Tidak), model akan mengarahkan kita ke jalur (cabang) berikutnya, menuju pertanyaan selanjutnya.
*   Proses ini berlanjut hingga kita mencapai kotak terakhir (disebut 'leaf node' atau daun), yang merupakan **keputusan akhir atau prediksi** model: 'Pasien Terindikasi Diabetes' atau 'Pasien Tidak Terindikasi Diabetes'.
*   Jadi, untuk pasien baru, model hanya perlu mengikuti pertanyaan-pertanyaan ini berdasarkan data kesehatan mereka hingga mencapai daun dan memberikan prediksi.

## 9. Bagaimana Metrik Kinerja (khususnya Recall) Dihitung:

*   Setelah model membuat prediksi untuk sekelompok pasien (misalnya, pada data pengujian kami), kami perlu mengevaluasi seberapa akurat prediksi tersebut. Salah satu metrik kunci yang kami gunakan adalah **Recall**.
*   **Recall berfokus pada seberapa baik model kami mengidentifikasi kasus positif yang sebenarnya.**
    *   Kami menghitung berapa banyak pasien yang **benar-benar memiliki diabetes** DAN model kami **berhasil memprediksi mereka memiliki diabetes**. Ini kami sebut **'True Positives' (TP)**.
    *   Kami juga menghitung berapa banyak pasien yang **benar-benar memiliki diabetes** tetapi model kami **salah memprediksi mereka TIDAK memiliki diabetes**. Ini kami sebut **'False Negatives' (FN)**.
*   **Rumusnya sederhana:**
    `Recall = True Positives / (True Positives + False Negatives)`
*   **Apa artinya?** Recall memberi tahu kita: 'Dari semua pasien yang *sebenarnya* menderita diabetes, berapa persen yang berhasil diidentifikasi dengan benar oleh model kami?' Skor Recall yang tinggi sangat penting dalam diagnosis medis karena kita ingin meminimalkan kasus diabetes yang terlewatkan.

## 10. Bagaimana *Cross-Validation* Bekerja (untuk 'Rata-rata Recall (Cross-Validation)'):

*   Ketika kami menyebut 'Rata-rata Recall (Cross-Validation)' untuk model baseline, ini adalah cara yang lebih kuat untuk memperkirakan kinerja model pada data pelatihan.
*   Alih-alih hanya melatih model sekali, kami membagi data pelatihan kami menjadi beberapa bagian (misalnya, 5 bagian atau 'folds').
*   Kemudian, kami melatih model sebanyak 5 kali. Setiap kali, kami menggunakan satu bagian sebagai 'data validasi sementara' dan 4 bagian lainnya untuk melatih model. Kami menghitung Recall untuk setiap dari 5 kali pelatihan/validasi ini.
*   Akhirnya, kami mengambil **rata-rata dari 5 skor Recall** tersebut. Rata-rata ini memberikan estimasi kinerja model yang lebih stabil dan handal selama fase pelatihannya, mengurangi dampak dari pembagian data tunggal.
