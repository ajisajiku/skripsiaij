import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap
import io
import contextlib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def train_baseline_model(X_train_scaled, y_train):
    base_model = DecisionTreeClassifier(random_state=42)
    base_model.fit(X_train_scaled, y_train)
    return base_model

def train_randomized_search_model(X_train_scaled, y_train):
    param_dist = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(1, 21)), 'min_samples_leaf': list(range(1, 51))}
    rand_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_dist, n_iter=100, cv=5, scoring='recall', random_state=42, n_jobs=-1)
    
    rand_search.fit(X_train_scaled, y_train)

    # Generate a structured log from cv_results_
    random_search_log = """
### Ringkasan RandomizedSearchCV

**Parameter Terbaik:**
{}

**Skor Terbaik (Recall):**
{:.4f}

**Top 5 Hasil (berdasarkan skor recall):**
""".format(rand_search.best_params_, rand_search.best_score_)

    results_df = pd.DataFrame(rand_search.cv_results_)
    results_df = results_df.sort_values(by='mean_test_score', ascending=False)
    
    for i, row in results_df.head(5).iterrows():
        random_search_log += "- Parameter: {} | Skor Recall: {:.4f}\n".format(row['params'], row['mean_test_score'])
    
    return rand_search.best_estimator_, rand_search.best_params_, random_search_log

def evaluate_models(y_test, X_test_scaled, base_model, gwo_model, rand_model, feature_names):
    st.subheader('Tabel Perbandingan Metrik')

    # Lakukan prediksi
    y_pred_base = base_model.predict(X_test_scaled)
    y_pred_gwo = gwo_model.predict(X_test_scaled)
    y_pred_rand = rand_model.predict(X_test_scaled)

    # Buat dictionary untuk menyimpan metrik
    data = {
        'Model': ['Baseline DT', 'GWO-Optimized DT', 'RandomizedSearchCV DT'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_base),
            accuracy_score(y_test, y_pred_gwo),
            accuracy_score(y_test, y_pred_rand)
        ],
        'Precision': [
            precision_score(y_test, y_pred_base),
            precision_score(y_test, y_pred_gwo),
            precision_score(y_test, y_pred_rand)
        ],
        'Recall': [
            recall_score(y_test, y_pred_base),
            recall_score(y_test, y_pred_gwo),
            recall_score(y_test, y_pred_rand)
        ],
        'F1-Score': [
            f1_score(y_test, y_pred_base),
            f1_score(y_test, y_pred_gwo),
            f1_score(y_test, y_pred_rand)
        ]
    }

    comparison_df = pd.DataFrame(data).set_index('Model')
    st.dataframe(comparison_df)

    st.subheader('Analisis Confusion Matrix')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('Baseline Model')
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(base_model, X_test_scaled, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)

    with col2:
        st.write('GWO Model')
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(gwo_model, X_test_scaled, y_test, ax=ax, cmap='Greens')
        st.pyplot(fig)

    with col3:
        st.write('RandomizedSearch Model')
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(rand_model, X_test_scaled, y_test, ax=ax, cmap='Oranges')
        st.pyplot(fig)

    st.subheader('Visualisasi Pohon Keputusan')
    col1, col2 = st.columns(2)
    with col1:
        col1.write('Baseline Model (Overfit)')
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        plot_tree(base_model, feature_names=feature_names, filled=True, ax=ax1, max_depth=5, fontsize=8)
        col1.pyplot(fig1, clear_figure=True)

    with col2:
        col2.write('GWO-Optimized Model (Simple)')
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        plot_tree(gwo_model, feature_names=feature_names, filled=True, ax=ax2, max_depth=5, fontsize=8)
        col2.pyplot(fig2, clear_figure=True)

    st.caption('Catatan: Kedua pohon dibatasi hingga kedalaman 5 (max_depth=5) agar dapat dibaca.')

def perform_data_preprocessing(df_input):
    st.subheader('Langkah Preprocessing:')

    # 1. Mengganti nilai 0 dengan NaN di kolom-kolom tertentu.
    st.info("1. Mengganti nilai 0 dengan NaN di kolom-kolom tertentu (Glucose, BloodPressure, SkinThickness, Insulin, BMI)...")
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_replace:
        df_input[col] = df_input[col].replace(0, np.nan)
    st.success("1. Nilai 0 berhasil diganti dengan NaN.")
    st.write("Contoh Data setelah penggantian 0 dengan NaN:")
    st.dataframe(df_input.head())

    # 2. Memisahkan fitur (X) dan target (y).
    st.info("2. Memisahkan fitur (X) dan target (y)...")
    X = df_input.drop('Outcome', axis=1)
    y = df_input['Outcome']
    st.success("2. Fitur dan target berhasil dipisahkan.")
    st.write("Contoh Fitur (X) setelah pemisahan:")
    st.dataframe(X.head())
    st.write("Contoh Target (y) setelah pemisahan:")
    st.dataframe(y.head())

    # 3. Membagi data menjadi set pelatihan dan pengujian (80:20).
    st.info("3. Membagi data menjadi set pelatihan dan pengujian (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.success("3. Data berhasil dibagi menjadi set pelatihan dan pengujian.")
    st.write("Contoh Data Pelatihan (X_train):")
    st.dataframe(X_train.head())
    st.write("Contoh Data Pengujian (X_test):")
    st.dataframe(X_test.head())

    # 4. Melakukan imputasi nilai NaN dengan median (fit pada data latih).
    st.info("4. Melakukan imputasi nilai NaN dengan median...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train = pd.DataFrame(X_train_imputed, columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X.columns)
    st.success("4. Imputasi nilai NaN berhasil dilakukan.")
    st.write("Contoh Data Pelatihan (X_train) setelah imputasi:")
    st.dataframe(X_train.head())

    # 5. Melakukan penskalaan fitur menggunakan StandardScaler (fit pada data latih).
    st.info("5. Melakukan penskalaan fitur menggunakan StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    feature_names = X.columns.tolist()
    st.success("5. Penskalaan fitur berhasil dilakukan.")
    st.write("Contoh Data Pelatihan (X_train) setelah penskalaan:")
    st.dataframe(X_train_scaled.head())

    st.success('Preprocessing Selesai!')
    st.write("Data siap untuk optimasi model.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer, feature_names