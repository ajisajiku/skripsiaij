import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def interactive_prediction_module(gwo_model, imputer, scaler, feature_names, X_train_scaled):
    st.info('Silakan masukkan 8 parameter pasien di bawah ini.')

    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            preg = col1.number_input('Pregnancies', min_value=0, step=1)
            gluc = col1.number_input('Glucose', min_value=0, step=1)
            bp = col1.number_input('BloodPressure', min_value=0, step=1)
            skin = col1.number_input('SkinThickness', min_value=0, step=1)

        with col2:
            ins = col2.number_input('Insulin', min_value=0, step=1)
            bmi = col2.number_input('BMI', min_value=0.0, format='%.1f')
            dpf = col2.number_input('DiabetesPedigreeFunction', min_value=0.0, format='%.3f')
            age = col2.number_input('Age', min_value=0, step=1)

        submit_button = st.form_submit_button(label='Dapatkan Prediksi', type='primary')

    if submit_button:
        user_input = [preg, gluc, bp, skin, ins, bmi, dpf, age]
        input_df = pd.DataFrame([user_input], columns=feature_names)

        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        input_df[cols_with_zeros] = input_df[cols_with_zeros].replace(0, np.nan)

        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        prediction = gwo_model.predict(input_scaled)
        probabilities = gwo_model.predict_proba(input_scaled)

        st.subheader('Hasil Prediksi')
        if prediction[0] == 1:
            st.error('Pasien TERINDIKASI Diabetes', icon='⚠️')
            confidence = probabilities[0][1]
        else:
            st.success('Pasien TIDAK TERINDIKASI Diabetes', icon='✅')
            confidence = probabilities[0][0]
        st.metric('Tingkat Keyakinan', f'{confidence*100:.2f}%')

        st.subheader('Penjelasan Prediksi (XAI)')
        st.write('Plot Bar ini menunjukkan kontribusi setiap fitur untuk prediksi.')
        st.write("Fitur dengan bar merah mendorong prediksi ke arah 'Diabetes', fitur dengan bar biru mendorong ke arah 'Tidak Diabetes'.")

        with st.spinner('Membuat plot penjelasan...'):
            try:
                explainer = shap.TreeExplainer(gwo_model, X_train_scaled)
                shap_values_list = explainer.shap_values(input_scaled)
                shap_values_single = shap_values_list[0][0] 
                
                fig, ax = plt.subplots()
                shap.bar_plot(shap_values_single, features=input_scaled[0], feature_names=feature_names, show=False)
                st.pyplot(fig, clear_figure=True)
            
            except Exception as e:
                st.error(f"Gagal membuat plot SHAP: {e}")
                st.error("Ini adalah bug yang sulit di-debug. Silakan coba input lain atau restart aplikasi.")
