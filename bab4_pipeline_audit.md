# Audit Pipeline Preprocessing Aktual Aplikasi

Tujuan audit ini adalah memastikan urutan preprocessing yang ditulis pada Bab 4 sama persis dengan urutan yang dijalankan program.

## Fungsi Preprocessing Utama

Fungsi preprocessing utama yang dipanggil sebelum training model adalah `perform_data_preprocessing(df_input)` pada `model_utils.py`. Fungsi ini dipanggil dari `app.py` ketika pengguna menekan tombol `Mulai Preprocessing Data`.

Potongan pemanggilan dari `app.py`:

```python
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
```

Fungsi lengkap dari `model_utils.py`:

```python
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
```

## Urutan Proses Aktual

Langkah 1: Dataset diunggah melalui Streamlit, lalu dibaca dengan `pd.read_csv(uploaded_file)`.

Langkah 2: Nilai `0` pada atribut `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diganti menjadi `NaN`.

Langkah 3: Dataset dipisahkan menjadi fitur `X` dan target `y`. Fitur adalah seluruh kolom selain `Outcome`, sedangkan target adalah kolom `Outcome`.

Langkah 4: Data dibagi menjadi data training dan data testing menggunakan `train_test_split(X, y, test_size=0.2, random_state=42)`.

Langkah 5: Imputasi median dilakukan setelah split. Objek `SimpleImputer(strategy='median')` di-fit hanya pada `X_train`, lalu digunakan untuk transformasi `X_train` dan `X_test`.

Langkah 6: Standardisasi dilakukan setelah imputasi. Objek `StandardScaler()` di-fit hanya pada `X_train` hasil imputasi, lalu digunakan untuk transformasi `X_train` dan `X_test`.

Langkah 7: Hasil preprocessing dikembalikan sebagai `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`, `scaler`, `imputer`, dan `feature_names`.

Langkah 8: Training model baseline, RandomizedSearchCV, dan GWO menggunakan `X_train_scaled` dan `y_train`.

## Jawaban Urutan Sebenarnya

| Pertanyaan | Jawaban |
| --- | --- |
| Apakah `train_test_split` dilakukan sebelum imputasi? | Ya. Split dilakukan lebih dulu pada `X` dan `y`, baru setelah itu imputasi median dilakukan pada `X_train` dan `X_test`. |
| Apakah imputasi dilakukan sebelum `train_test_split`? | Tidak. Imputasi tidak dilakukan pada seluruh dataset sebelum split. |
| Apakah `StandardScaler` dilakukan sebelum split? | Tidak. `StandardScaler` tidak di-fit pada seluruh dataset sebelum split. |
| Apakah `StandardScaler` dilakukan sesudah split? | Ya. `StandardScaler` dilakukan setelah split dan setelah imputasi median. Scaler di-fit pada data training hasil imputasi, kemudian digunakan untuk transformasi data training dan testing. |

## Potongan Kode Bukti Urutan

Nilai 0 diganti menjadi `NaN` sebelum split:

```python
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_replace:
    df_input[col] = df_input[col].replace(0, np.nan)
```

Fitur dan target dipisahkan:

```python
X = df_input.drop('Outcome', axis=1)
y = df_input['Outcome']
```

Split dilakukan sebelum imputasi:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Imputasi median dilakukan setelah split, dengan `fit_transform` hanya pada data training:

```python
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

StandardScaler dilakukan setelah imputasi, dengan `fit_transform` hanya pada data training:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Model menerima data hasil preprocessing:

```python
X_train = st.session_state['X_train_scaled']
y_train = st.session_state['y_train']
```

Baseline Decision Tree dilatih dengan data training yang sudah diskalakan:

```python
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train, y_train)
```

## Diagram Pipeline Aktual

```text
Dataset CSV
↓
Upload dan baca dataset dengan pd.read_csv
↓
Ganti nilai 0 menjadi NaN
(Glucose, BloodPressure, SkinThickness, Insulin, BMI)
↓
Pisahkan fitur X dan target y
↓
Train-test split 80:20
(test_size=0.2, random_state=42)
↓
Fit SimpleImputer(strategy='median') pada X_train
↓
Transform X_train dan X_test menggunakan imputer dari X_train
↓
Fit StandardScaler pada X_train hasil imputasi
↓
Transform X_train dan X_test menggunakan scaler dari X_train
↓
X_train_scaled, X_test_scaled, y_train, y_test
↓
Training model
(Decision Tree baseline / RandomizedSearchCV / GWO)
↓
Evaluasi model pada X_test_scaled dan y_test
```

## Komponen Preprocessing dan Sampling

| Komponen | Ada di pipeline utama aplikasi? | Bukti |
| --- | --- | --- |
| `SimpleImputer` | Ada | `imputer = SimpleImputer(strategy='median')` |
| `StandardScaler` | Ada | `scaler = StandardScaler()` |
| `MinMaxScaler` | Tidak ada pada pipeline utama aplikasi | Import preprocessing utama hanya menggunakan `StandardScaler` pada `model_utils.py` |
| `SMOTE` | Tidak ada | Tidak ditemukan import atau pemanggilan `SMOTE` pada file Python utama |
| Undersampling | Tidak ada | Tidak ditemukan proses undersampling pada pipeline utama |
| Oversampling | Tidak ada | Tidak ditemukan proses oversampling pada pipeline utama |

## Catatan Audit Project

File Python utama yang mereplikasi pipeline, seperti `comparison.py` dan `diabetes_prediction.py`, juga menggunakan urutan yang sama secara garis besar: split terlebih dahulu, imputasi median setelah split, lalu `StandardScaler` setelah imputasi.

`MinMaxScaler` tidak digunakan pada pipeline utama aplikasi. Jika ada pembahasan Bab 4 yang menyebut `MinMaxScaler`, bagian tersebut perlu disesuaikan menjadi `StandardScaler` agar konsisten dengan kode aplikasi.

## Kesimpulan

Urutan preprocessing aktual aplikasi adalah:

```text
Ganti 0 menjadi NaN
→ Pisahkan X dan y
→ Train-test split
→ Imputasi median fit pada X_train
→ Transform X_train dan X_test
→ StandardScaler fit pada X_train hasil imputasi
→ Transform X_train dan X_test
→ Training model
```

Dengan demikian, Bab 4 harus menuliskan bahwa `train_test_split` dilakukan sebelum imputasi median dan sebelum `StandardScaler`. Imputasi dan scaling tidak dilakukan pada seluruh dataset sebelum split, sehingga pipeline ini sudah menghindari data leakage dari data testing ke proses preprocessing.
