
# BAB 5: KESIMPULAN DAN SARAN

Bab ini merangkum temuan-temuan utama dari penelitian, menyajikan insight praktis yang dapat ditarik, serta memberikan saran untuk pengembangan di masa depan.

## 5.1 Kesimpulan

Berdasarkan penelitian dan pengembangan yang telah dilakukan, dapat ditarik beberapa kesimpulan utama:

1.  **Keberhasilan Optimisasi GWO:** Algoritma *Grey Wolf Optimizer* (GWO) terbukti **berhasil dan efektif** dalam mengoptimalkan hyperparameter model klasifikasi *Decision Tree*. Tujuan utama optimisasi untuk **memaksimalkan nilai Recall** tercapai, di mana model GWO-DT menghasilkan **Recall sebesar 0.764**. 

2.  **Keunggulan Model GWO-DT:** Dibandingkan dengan model *Baseline Decision Tree* (Recall 0.655) dan model yang dioptimalkan dengan *RandomizedSearchCV* (Recall 0.691), model GWO-DT menunjukkan **kemampuan superior dalam mengidentifikasi kasus positif diabetes**. Hal ini sangat krusial dalam konteks medis untuk meminimalkan risiko *false negative* (pasien diabetes tidak terdeteksi).

3.  **Implementasi Aplikasi Interaktif:** Penelitian ini berhasil menghasilkan sebuah **prototipe aplikasi web interaktif** menggunakan Streamlit. Aplikasi ini tidak hanya menjalankan proses pemodelan tetapi juga menyediakan antarmuka bagi pengguna untuk melakukan prediksi secara langsung pada data pasien baru, serta memahami faktor-faktor penentu prediksi melalui visualisasi SHAP.

4.  **Transparansi Model:** Integrasi *SHAP (SHapley Additive exPlanations)* dalam aplikasi memberikan **transparansi dan interpretabilitas** terhadap prediksi yang dihasilkan model. Ini memungkinkan pemangku kepentingan (seperti praktisi medis) untuk memahami fitur apa yang paling berpengaruh dalam sebuah keputusan, sehingga meningkatkan kepercayaan terhadap model.

## 5.2 Insight Praktis

Temuan dari penelitian ini menawarkan beberapa insight praktis:

-   **Alat Bantu Skrining Awal:** Aplikasi yang dikembangkan dapat berfungsi sebagai alat bantu skrining (penapisan) awal yang efektif bagi tenaga medis. Fokus pada recall yang tinggi memastikan lebih banyak kasus potensial dapat terjaring untuk pemeriksaan lebih lanjut.
-   **Otomatisasi Tuning:** Penggunaan GWO menunjukkan potensi algoritma metaheuristik sebagai alternatif yang kuat selain metode standar seperti *Randomized Search* untuk otomatisasi *hyperparameter tuning*, terutama ketika target evaluasi spesifik (seperti recall) menjadi prioritas.

## 5.3 Saran Pengembangan

Untuk pengembangan di masa depan, beberapa area berikut dapat dieksplorasi untuk meningkatkan kualitas dan kegunaan sistem:

1.  **Eksplorasi Algoritma Lain:** Mencoba algoritma klasifikasi yang lebih kompleks seperti **XGBoost, LightGBM, atau Random Forest** dan mengoptimalkannya dengan GWO untuk melihat apakah dapat dicapai keseimbangan yang lebih baik antara Recall dan metrik lainnya (seperti Precision dan Accuracy).

2.  **Pengembangan Fitur Aplikasi:**
    -   **Deploy ke Cloud:** Menyebarkan (deploy) aplikasi Streamlit ke platform cloud (misalnya Streamlit Community Cloud, Heroku) agar dapat diakses secara luas untuk pengujian dan pengumpulan umpan balik.
    -   **Penjelasan Prediksi Individual:** Melengkapi modul prediksi interaktif dengan visualisasi SHAP *lokal* (seperti *force plot* atau *waterfall plot*) untuk menjelaskan faktor-faktor yang memengaruhi prediksi **untuk satu pasien spesifik**.

3.  **Peningkatan Kualitas Data:** Menggunakan dataset yang **lebih besar dan lebih beragam** dari berbagai populasi untuk melatih model. Hal ini dapat meningkatkan kemampuan generalisasi dan keadilan (fairness) model terhadap demografi yang berbeda.

4.  **Integrasi Sistem:** Mengkaji kemungkinan untuk mengintegrasikan model prediksi ini dengan **Sistem Informasi Rumah Sakit (SIRS)** atau rekam medis elektronik (EHR), sehingga proses prediksi dapat berjalan secara lebih otomatis berdasarkan data pasien yang sudah ada.
