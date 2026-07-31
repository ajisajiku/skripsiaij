import sys
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import random
import matplotlib.pyplot as plt
import shap
import io
import contextlib
from gwo_optimizer import calculate_fitness, run_gwo

from model_utils import train_baseline_model, train_randomized_search_model, evaluate_models, perform_data_preprocessing
from interactive_prediction import interactive_prediction_module

# 2. Atur konfigurasi halaman
st.set_page_config(page_title='Optimasi GWO-DT', layout='wide')

# 3. Buat judul utama
st.title('Optimasi Decision Tree dengan Grey Wolf Optimizer (GWO) untuk Prediksi Diabetes')

# 4. Buat navigasi di sidebar
st.sidebar.title('Navigasi Modul')

# Initialize session state variables if they don't exist
if 'data_preprocessed' not in st.session_state:
    st.session_state['data_preprocessed'] = False
if 'X_train_scaled' not in st.session_state:
    st.session_state['X_train_scaled'] = None
if 'X_test_scaled' not in st.session_state:
    st.session_state['X_test_scaled'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'imputer' not in st.session_state:
    st.session_state['imputer'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'gwo_best_params' not in st.session_state:
    st.session_state['gwo_best_params'] = None
if 'gwo_best_score' not in st.session_state:
    st.session_state['gwo_best_score'] = None
if 'base_model' not in st.session_state:
    st.session_state['base_model'] = None
if 'gwo_model' not in st.session_state:
    st.session_state['gwo_model'] = None
if 'rand_model' not in st.session_state:
    st.session_state['rand_model'] = None
if 'rand_best_params' not in st.session_state:
    st.session_state['rand_best_params'] = None
if 'rand_log' not in st.session_state:
    st.session_state['rand_log'] = None
if 'gwo_score_history' not in st.session_state:
    st.session_state['gwo_score_history'] = None





# 5. Gunakan 'st.sidebar.radio' untuk membuat 4 pilihan halaman
page = st.sidebar.radio(
    "Pilih Modul:",
    [
        '1. Upload & Preprocessing',
        '2. Optimasi Model',
        '3. Evaluasi & Perbandingan',
        '4. Prediksi Interaktif'
    ]
)

# 6. Buat struktur 'if/elif' untuk 4 halaman tersebut
if page == '1. Upload & Preprocessing':
    st.header('Modul 1: Upload & Preprocessing Data')
    st.info('Silakan upload dataset Pima Indians Diabetes (format CSV).')

    uploaded_file = st.file_uploader('Pilih file CSV', type=['csv'])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.subheader('Data Awal (Head)')
        st.dataframe(df_input.head())

        process_button = st.button('Mulai Preprocessing Data', type='primary')

        if process_button:


            X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer, feature_names = perform_data_preprocessing(df_input)

            # Store preprocessed data and objects in session state
            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['y_train'] = y_train
            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['y_test'] = y_test
            st.session_state['scaler'] = scaler
            st.session_state['imputer'] = imputer
            st.session_state['feature_names'] = feature_names
            st.session_state['data_preprocessed'] = True

elif page == '2. Optimasi Model':
    st.header('Modul 2: Optimasi Model')

    if 'X_train_scaled' not in st.session_state or not st.session_state['data_preprocessed']:
        st.warning('Harap upload dan proses data di Halaman 1 terlebih dahulu.')
        st.stop()



    st.info('Data siap. Pilih salah satu tab di bawah untuk memulai proses optimasi.')

    tab1, tab2, tab3 = st.tabs(['1️⃣ Model Baseline', '2️⃣ Optimasi RandomizedSearch', '3️⃣ Optimasi GWO'])

    X_train = st.session_state['X_train_scaled']
    y_train = st.session_state['y_train']

    with tab1:
        st.subheader('Latih Model Baseline')
        

        
        # Tombol untuk melatih
        if st.button('Latih Model Baseline'):
            with st.spinner('Melatih Model Baseline...'):
                st.info("Memulai pelatihan Decision Tree Model Baseline...")
                base_model = DecisionTreeClassifier(random_state=42)
                st.write("Parameter default Decision Tree:")
                st.json(base_model.get_params()) # Display default parameters

                st.info("Melatih model menggunakan data pelatihan...")
                base_model.fit(X_train, y_train)
                
                # Calculate cross-validation recall score
                cv_scores = cross_val_score(base_model, X_train, y_train, cv=5, scoring='recall')
                avg_recall = np.mean(cv_scores)
                
                st.success('Selesai. Model Baseline disimpan.')
                st.session_state['base_model'] = base_model

                st.markdown('---')
                st.write('**Model Baseline Telah Dilatih.**')
                st.write(f"- Rata-rata Recall (Cross-Validation): {avg_recall:.4f}")
                st.write('Berikut adalah beberapa karakteristik model yang dilatih:')
                st.write(f"- Kedalaman Pohon (Max Depth): {base_model.tree_.max_depth}")
                st.write(f"- Jumlah Daun (Leaf Nodes): {base_model.tree_.n_leaves}")

                st.info("Visualisasi sebagian dari Pohon Keputusan Model Baseline (kedalaman terbatas untuk keterbacaan): ")
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(base_model, feature_names=st.session_state['feature_names'], filled=True, ax=ax, max_depth=3, fontsize=8)
                st.pyplot(fig, clear_figure=True)

        # Tampilkan hasil jika model SUDAH dilatih
        if st.session_state['base_model'] is not None:
            st.markdown('---')
            st.write('**Hasil Pelatihan Baseline (dari sesi sebelumnya):**')
            st.success('✅ Model Baseline tersedia.')
            
            with st.expander('Klik untuk melihat apa yang terjadi pada Model Baseline'):
                st.markdown("""
                Model 'Baseline' adalah Decision Tree standar tanpa batasan.
                
                **Apa yang baru saja terjadi:**
                1.  Model dilatih dengan parameter default, seperti yang terlihat di atas.
                2.  Visualisasi pohon menunjukkan bagaimana model membuat keputusan berdasarkan fitur-fitur.
                3.  Karena model ini menggunakan algoritma 'rakus' (greedy) dan tidak memiliki batasan kedalaman atau jumlah daun (kecuali yang diatur secara default oleh scikit-learn), ia cenderung tumbuh sangat dalam.
                4.  Pohon yang dalam dan kompleks seperti ini rentan terhadap *overfitting*, yaitu terlalu cocok dengan data pelatihan dan mungkin tidak bekerja dengan baik pada data baru.
                """)

    with tab2:

            st.subheader('Optimasi Model dengan RandomizedSearch')

            

            # Tampilkan hasil JIKA SUDAH dilatih

            if st.session_state['rand_model'] is not None:

                st.markdown('---')

                st.write('**Hasil Optimasi RandomizedSearch:**')

                st.success(f"✅ Selesai. Parameter terbaik: {st.session_state['rand_best_params']}")
                st.caption("Catatan: 'Skor Terbaik (Recall)' di log di bawah adalah skor cross-validation pada data pelatihan.")

                

                with st.expander('Klik untuk melihat Log Mentah RandomizedSearchCV (Bukti Pencarian)'):

                    st.code(st.session_state['rand_log'])

            if st.session_state['rand_model'] is None:
                st.info('Model ini belum dilatih. Tekan tombol di bawah untuk memulai.')
                if st.button('Latih RandomizedSearch'):
                    with st.status("Menjalankan RandomizedSearchCV... Ini mungkin memakan waktu...", expanded=True) as status:
                        st.write("Memulai optimasi model dengan RandomizedSearchCV...")

                        rand_model, rand_best_params, random_search_log = train_randomized_search_model(X_train, y_train)

                        st.code(random_search_log)

                        st.session_state['rand_model'] = rand_model
                        st.session_state['rand_log'] = random_search_log
                        st.session_state['rand_best_params'] = rand_best_params
                        status.update(label="Optimasi RandomizedSearchCV Selesai!", state="complete", expanded=False)
                        st.success('Selesai. Model RandomizedSearch disimpan.')
                        st.rerun() # Rerun to update the UI and hide the button

    with tab3:
        st.subheader('Optimasi Model dengan GWO')

        # Tampilkan hasil JIKA SUDAH dilatih
        if 'gwo_params' in st.session_state:
            st.markdown('---')
            st.success(f"✅ Selesai. Model GWO disimpan dengan parameter: {st.session_state['gwo_params']}")
            
            st.markdown('**Log Optimasi Final:**')
            st.markdown(st.session_state['gwo_final_log_text'])
            st.caption("Catatan: 'Best Score' di log di atas adalah skor cross-validation pada data pelatihan.")
            st.markdown('**Plot Konvergensi Final:**')
            st.line_chart(st.session_state['gwo_score_history'])
            
        # Tampilkan tombol HANYA JIKA BELUM dilatih
        else:
            st.info('Model ini belum dilatih. Tekan tombol di bawah untuk memulai.')
            if st.button('Mulai Optimasi GWO'):
                with st.spinner('Menjalankan GWO...'):
                    gwo_history = []
                    log_placeholder = st.empty()
                    plot_placeholder = st.empty()
                    
                    # Ambil data latih
                    X_train_scaled = st.session_state['X_train_scaled']
                    y_train = st.session_state['y_train']
                    
                    LOWER_BOUND, UPPER_BOUND, DIMENSIONS, NUM_WOLVES, MAX_ITERATIONS = [1, 1], [20, 50], 2, 10, 20

                    for update in run_gwo(calculate_fitness, LOWER_BOUND, UPPER_BOUND, DIMENSIONS, NUM_WOLVES, MAX_ITERATIONS, X_train_scaled, y_train):
                        gwo_history.append(update['alpha_score'])
                        log_text = f"**Iterasi:** {update['iteration']}/{MAX_ITERATIONS} | **Best Score:** `{update['alpha_score']:.4f}`"
                        log_placeholder.markdown(log_text)
                        plot_placeholder.line_chart(gwo_history)
                        last_gwo_update = update
                        
                    # Simpan model GWO
                    best_params = last_gwo_update['alpha_params']
                    gwo_depth = int(best_params[0])
                    gwo_leaf = int(best_params[1])
                    gwo_model = DecisionTreeClassifier(max_depth=gwo_depth, min_samples_leaf=gwo_leaf, criterion='gini', random_state=42)
                    gwo_model.fit(X_train_scaled, y_train)
                    
                    # Simpan semua hasil ke session_state
                    st.session_state['gwo_model'] = gwo_model
                    st.session_state['gwo_params'] = f'Depth={gwo_depth}, Leaf={gwo_leaf}'
                    st.session_state['gwo_final_log_text'] = log_text
                    st.session_state['gwo_score_history'] = gwo_history
                    
                    # Rerun untuk menampilkan hasil
                    st.rerun()

            





elif page == '3. Evaluasi & Perbandingan':
    st.header('Modul 3: Evaluasi & Perbandingan Model')
    if 'gwo_model' not in st.session_state:
        st.warning('Harap jalankan optimasi di Halaman 2 terlebih dahulu.')
        st.stop()
    
    y_test = st.session_state['y_test']
    X_test_scaled = st.session_state['X_test_scaled']
    base_model = st.session_state['base_model']
    gwo_model = st.session_state['gwo_model']
    rand_model = st.session_state['rand_model']
    feature_names = st.session_state['feature_names']

    evaluate_models(y_test, X_test_scaled, base_model, gwo_model, rand_model, feature_names)



elif page == '4. Prediksi Interaktif':
    st.header('Modul 4: Prediksi Pasien Baru (Interaktif')
    if 'gwo_model' not in st.session_state or 'imputer' not in st.session_state or 'scaler' not in st.session_state:
        st.warning('Harap jalankan optimasi di Halaman 2 terlebih dahulu untuk memuat model dan preprocessor.')
        st.stop()
    
    gwo_model = st.session_state['gwo_model']
    imputer = st.session_state['imputer']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    X_train_scaled = st.session_state['X_train_scaled']

    interactive_prediction_module(gwo_model, imputer, scaler, feature_names, X_train_scaled)


st.sidebar.info("Pilih modul di atas untuk navigasi.")
