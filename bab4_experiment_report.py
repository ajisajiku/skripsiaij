import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
DATASET_PATH = Path("diabetes.csv")
OUTPUT_MD = Path("bab4_experiment_report_output.md")
OUTPUT_TSV = Path("bab4_experiment_tables.tsv")
MISSING_VALUE_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
GWO_LOWER_BOUND = np.array([1, 1])
GWO_UPPER_BOUND = np.array([20, 50])
GWO_DIMENSIONS = 2
GWO_NUM_WOLVES = 10
GWO_MAX_ITERATIONS = 50


def md_table(df):
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(format_cell)

    headers = [str(col) for col in table.columns]
    separator = ["---"] * len(headers)
    rows = table.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def tsv_table(title, df):
    return f"\n## {title}\n" + df.to_csv(sep="\t", index=False)


def class_label(value):
    return "Diabetes" if int(value) == 1 else "Tidak Diabetes"


def calculate_manual_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall)
        else 0
    )
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


def fitness_gwo(params, X_train, y_train):
    max_depth = int(params[0])
    min_samples_leaf = int(params[1])
    if max_depth < 1 or min_samples_leaf < 1:
        return 1.0

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion="gini",
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")
    return 1.0 - float(np.mean(scores))


def run_gwo_experiment(X_train, y_train):
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    alpha_pos = np.zeros(GWO_DIMENSIONS)
    alpha_score = float("inf")
    beta_pos = np.zeros(GWO_DIMENSIONS)
    beta_score = float("inf")
    delta_pos = np.zeros(GWO_DIMENSIONS)
    delta_score = float("inf")

    positions = (
        np.random.uniform(0, 1, (GWO_NUM_WOLVES, GWO_DIMENSIONS))
        * (GWO_UPPER_BOUND - GWO_LOWER_BOUND)
        + GWO_LOWER_BOUND
    )

    history = []
    for iteration in range(GWO_MAX_ITERATIONS):
        for i in range(GWO_NUM_WOLVES):
            positions[i, :] = np.clip(positions[i, :], GWO_LOWER_BOUND, GWO_UPPER_BOUND)
            fitness = fitness_gwo(positions[i, :], X_train, y_train)

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif alpha_score < fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif alpha_score < fitness and beta_score < fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        history.append(
            {
                "Iterasi": iteration + 1,
                "Best Fitness (1 - Recall)": alpha_score,
                "Best Recall CV": 1.0 - alpha_score,
                "Max Depth": int(alpha_pos[0]),
                "Min Samples Leaf": int(alpha_pos[1]),
            }
        )

        a = 2 - iteration * (2 / GWO_MAX_ITERATIONS)
        for i in range(GWO_NUM_WOLVES):
            for j in range(GWO_DIMENSIONS):
                r1, r2 = random.random(), random.random()
                a1 = 2 * a * r1 - a
                c1 = 2 * r2
                d_alpha = abs(c1 * alpha_pos[j] - positions[i, j])
                x1 = alpha_pos[j] - a1 * d_alpha

                r1, r2 = random.random(), random.random()
                a2 = 2 * a * r1 - a
                c2 = 2 * r2
                d_beta = abs(c2 * beta_pos[j] - positions[i, j])
                x2 = beta_pos[j] - a2 * d_beta

                r1, r2 = random.random(), random.random()
                a3 = 2 * a * r1 - a
                c3 = 2 * r2
                d_delta = abs(c3 * delta_pos[j] - positions[i, j])
                x3 = delta_pos[j] - a3 * d_delta

                positions[i, j] = (x1 + x2 + x3) / 3

    best_params = {
        "criterion": "gini",
        "max_depth": int(alpha_pos[0]),
        "min_samples_leaf": int(alpha_pos[1]),
        "random_state": RANDOM_STATE,
    }
    return best_params, pd.DataFrame(history)


def add_section(parts, title, body):
    parts.append(f"\n## {title}\n\n{body}\n")


def main():
    df_raw = pd.read_csv(DATASET_PATH)
    df_zero_as_nan = df_raw.copy()
    df_zero_as_nan[MISSING_VALUE_COLUMNS] = df_zero_as_nan[MISSING_VALUE_COLUMNS].replace(
        0, np.nan
    )

    missing_counts = (
        df_zero_as_nan[MISSING_VALUE_COLUMNS]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "Atribut", 0: "Jumlah Missing Value"})
    )

    X = df_zero_as_nan.drop("Outcome", axis=1)
    y = df_zero_as_nan["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed_array = imputer.fit_transform(X_train)
    X_test_imputed_array = imputer.transform(X_test)
    X_train_imputed = pd.DataFrame(X_train_imputed_array, columns=X.columns)
    X_test_imputed = pd.DataFrame(X_test_imputed_array, columns=X.columns)

    medians = pd.DataFrame(
        {
            "Atribut": X.columns,
            "Median Data Training": imputer.statistics_,
        }
    )
    requested_medians = medians[medians["Atribut"].isin(MISSING_VALUE_COLUMNS)]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)

    manual_scaler_feature = "Glucose"
    manual_scaler_feature_index = X.columns.get_loc(manual_scaler_feature)
    manual_scaler_x = X_train_imputed.iloc[0, manual_scaler_feature_index]
    manual_scaler_mean = scaler.mean_[manual_scaler_feature_index]
    manual_scaler_std = scaler.scale_[manual_scaler_feature_index]
    manual_scaler_z = (manual_scaler_x - manual_scaler_mean) / manual_scaler_std
    scaler_formula_df = pd.DataFrame(
        [
            {
                "Metode": "StandardScaler",
                "Rumus": "z = (x - mean) / standard deviation",
                "Keterangan": "mean dan standard deviation dihitung dari data training",
            }
        ]
    )
    scaler_manual_df = pd.DataFrame(
        [
            {
                "Atribut": manual_scaler_feature,
                "Nilai x": manual_scaler_x,
                "Mean Training": manual_scaler_mean,
                "Standard Deviation Training": manual_scaler_std,
                "Perhitungan": (
                    f"({manual_scaler_x:.4f} - {manual_scaler_mean:.4f}) / "
                    f"{manual_scaler_std:.4f}"
                ),
                "Hasil z": manual_scaler_z,
                "Nilai dari StandardScaler": X_train_scaled.iloc[
                    0, manual_scaler_feature_index
                ],
            }
        ]
    )

    baseline_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_model.fit(X_train_scaled, y_train)

    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(1, 21)),
        "min_samples_leaf": list(range(1, 51)),
    }
    randomized_search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring="recall",
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=False,
    )
    randomized_search.fit(X_train_scaled, y_train)
    randomized_model = randomized_search.best_estimator_

    gwo_best_params, gwo_history = run_gwo_experiment(X_train_scaled, y_train)
    gwo_model = DecisionTreeClassifier(**gwo_best_params)
    gwo_model.fit(X_train_scaled, y_train)

    predictions = {
        "Baseline DT": baseline_model.predict(X_test_scaled),
        "RandomizedSearch DT": randomized_model.predict(X_test_scaled),
        "GWO DT": gwo_model.predict(X_test_scaled),
    }

    metrics_rows = []
    confusion_rows = []
    manual_rows = []
    for model_name, y_pred in predictions.items():
        manual = calculate_manual_metrics(y_test, y_pred)
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
            }
        )
        confusion_rows.append(
            {
                "Model": model_name,
                "TN": manual["TN"],
                "FP": manual["FP"],
                "FN": manual["FN"],
                "TP": manual["TP"],
            }
        )
        manual_rows.append(
            {
                "Model": model_name,
                "Accuracy Manual": (
                    f"({manual['TP']} + {manual['TN']}) / "
                    f"({manual['TP']} + {manual['TN']} + {manual['FP']} + {manual['FN']}) "
                    f"= {manual['Accuracy']:.4f}"
                ),
                "Precision Manual": (
                    f"{manual['TP']} / ({manual['TP']} + {manual['FP']}) "
                    f"= {manual['Precision']:.4f}"
                ),
                "Recall Manual": (
                    f"{manual['TP']} / ({manual['TP']} + {manual['FN']}) "
                    f"= {manual['Recall']:.4f}"
                ),
                "F1-Score Manual": (
                    f"2 * ({manual['Precision']:.4f} * {manual['Recall']:.4f}) / "
                    f"({manual['Precision']:.4f} + {manual['Recall']:.4f}) "
                    f"= {manual['F1-Score']:.4f}"
                ),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    manual_df = pd.DataFrame(manual_rows)

    test_examples = pd.DataFrame(
        {
            "Nomor": range(1, 11),
            "Nilai Aktual": y_test.iloc[:10].map(class_label).to_list(),
            "Prediksi Baseline": pd.Series(predictions["Baseline DT"][:10]).map(class_label),
            "Prediksi RandomizedSearch": pd.Series(
                predictions["RandomizedSearch DT"][:10]
            ).map(class_label),
            "Prediksi GWO": pd.Series(predictions["GWO DT"][:10]).map(class_label),
        }
    )

    rs_results = pd.DataFrame(randomized_search.cv_results_)
    rs_examples = rs_results[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].head(10)
    rs_examples = rs_examples.rename(
        columns={
            "params": "Kombinasi Parameter",
            "mean_test_score": "Mean Recall CV",
            "std_test_score": "Std Recall CV",
            "rank_test_score": "Rank",
        }
    )
    rs_examples["Kombinasi Parameter"] = rs_examples["Kombinasi Parameter"].astype(str)

    selected_gwo_iterations = gwo_history[
        gwo_history["Iterasi"].isin([1, 10, 20, 30, 40, 50])
    ]

    baseline_params = pd.DataFrame(
        [{"Parameter": key, "Nilai": str(value)} for key, value in baseline_model.get_params().items()]
    )
    rs_best_params = pd.DataFrame(
        [{"Parameter": key, "Nilai": str(value)} for key, value in randomized_search.best_params_.items()]
    )
    gwo_param_info = pd.DataFrame(
        [
            {"Parameter": "Jumlah wolf", "Nilai": GWO_NUM_WOLVES},
            {"Parameter": "Jumlah iterasi", "Nilai": GWO_MAX_ITERATIONS},
            {"Parameter": "Fungsi fitness", "Nilai": "1 - mean recall 5-fold CV"},
            {"Parameter": "Dimensi", "Nilai": GWO_DIMENSIONS},
            {"Parameter": "Lower bound", "Nilai": str(GWO_LOWER_BOUND.tolist())},
            {"Parameter": "Upper bound", "Nilai": str(GWO_UPPER_BOUND.tolist())},
        ]
    )
    gwo_best_params_df = pd.DataFrame(
        [{"Parameter": key, "Nilai": str(value)} for key, value in gwo_best_params.items()]
    )

    split_df = pd.DataFrame(
        [
            {"Subset": "Training", "Jumlah Data": len(X_train_scaled)},
            {"Subset": "Testing", "Jumlah Data": len(X_test_scaled)},
        ]
    )

    markdown_parts = [
        "# Laporan Eksperimen Bab 4\n",
        "Catatan project: `app.py` dan `model_utils.py` menggunakan `StandardScaler`. "
        "Script eksperimen Bab 4 ini sudah disesuaikan agar mengikuti pipeline utama aplikasi.\n",
    ]

    add_section(
        markdown_parts,
        "Struktur Project",
        "\n".join(
            [
                "- `app.py`: file utama aplikasi Streamlit dan navigasi modul.",
                "- `model_utils.py`: preprocessing, training baseline, RandomizedSearchCV, dan evaluasi UI.",
                "- `gwo_optimizer.py`: fungsi fitness dan generator proses Grey Wolf Optimizer.",
                "- `comparison.py`: script pembanding metrik baseline, GWO, dan RandomizedSearchCV.",
                "- `diabetes_prediction.py`: script training GWO dan pembuatan SHAP summary plot.",
                "- `interactive_prediction.py`: modul UI prediksi interaktif Streamlit.",
                "- `diabetes.csv`: dataset Pima Indians Diabetes.",
                "- `bab4_experiment_report.py`: script eksperimen Bab 4 terpisah.",
            ]
        ),
    )
    add_section(
        markdown_parts,
        "Alur Program",
        "\n".join(
            [
                "1. Dataset dimuat dari `diabetes.csv`.",
                "2. Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diperlakukan sebagai missing value, lalu diganti menjadi NaN.",
                "3. Fitur dipisahkan dari target `Outcome`.",
                "4. Data dibagi menjadi training dan testing dengan rasio 80:20 menggunakan `random_state=42`.",
                "5. Missing value diimputasi memakai median yang dipelajari dari data training.",
                "6. Fitur distandardisasi memakai `StandardScaler` yang di-fit pada data training.",
                "7. Decision Tree baseline dilatih menggunakan parameter default scikit-learn dengan `random_state=42`.",
                "8. RandomizedSearchCV mencari kombinasi `criterion`, `max_depth`, dan `min_samples_leaf` dengan scoring recall.",
                "9. GWO mencari `max_depth` dan `min_samples_leaf` berdasarkan fitness `1 - mean recall 5-fold CV`.",
                "10. Model dievaluasi pada data testing menggunakan confusion matrix, accuracy, precision, recall, dan F1-score.",
            ]
        ),
    )

    tables = [
        ("10 Data Awal Sebelum Preprocessing", df_raw.head(10)),
        ("Missing Value Setelah Nilai 0 Dianggap Missing", missing_counts),
        ("Median Tiap Atribut", requested_medians),
        ("10 Data Training Setelah Imputasi Median", X_train_imputed.head(10)),
        ("10 Data Training Setelah StandardScaler", X_train_scaled.head(10)),
        ("Rumus StandardScaler", scaler_formula_df),
        ("Contoh Perhitungan Manual StandardScaler", scaler_manual_df),
        ("Jumlah Data Training dan Testing", split_df),
        ("Parameter Decision Tree Baseline", baseline_params),
        ("10 Contoh Data Testing dan Prediksi", test_examples),
        ("Confusion Matrix", confusion_df),
        ("Metrik Evaluasi", metrics_df),
        ("Perhitungan Manual Metrik", manual_df),
        ("Parameter Terbaik RandomizedSearch", rs_best_params),
        ("10 Contoh Iterasi RandomizedSearch", rs_examples),
        ("Parameter GWO", gwo_param_info),
        ("Perkembangan Fitness GWO", selected_gwo_iterations),
        ("Parameter Terbaik GWO", gwo_best_params_df),
    ]

    for title, table in tables:
        add_section(markdown_parts, title, md_table(table))

    add_section(
        markdown_parts,
        "Narasi Singkat Akademik",
        "\n".join(
            [
                "Data awal menunjukkan beberapa atribut klinis memiliki nilai 0 yang secara medis tidak wajar, sehingga nilai tersebut diperlakukan sebagai missing value.",
                "Imputasi median digunakan karena lebih stabil terhadap nilai ekstrem dibandingkan rata-rata.",
                "StandardScaler mengubah fitur menjadi nilai z-score dengan rumus z = (x - mean) / standard deviation, sehingga fitur berada pada skala yang lebih sebanding sesuai pipeline utama aplikasi.",
                "Pembagian data 80:20 menghasilkan data training untuk pembentukan model dan data testing untuk evaluasi generalisasi.",
                "Baseline Decision Tree digunakan sebagai pembanding awal tanpa optimasi hyperparameter.",
                "RandomizedSearchCV dan GWO sama-sama mengoptimasi hyperparameter dengan fokus pada recall agar model lebih sensitif dalam mengenali kelas diabetes.",
                "Evaluasi akhir menggunakan confusion matrix serta accuracy, precision, recall, dan F1-score untuk membandingkan performa model secara menyeluruh.",
            ]
        ),
    )

    report_text = "\n".join(markdown_parts)
    OUTPUT_MD.write_text(report_text, encoding="utf-8")

    tsv_text = "\n".join(tsv_table(title, table) for title, table in tables)
    OUTPUT_TSV.write_text(tsv_text, encoding="utf-8")

    print(report_text)
    print(f"\nFile Markdown dibuat: {OUTPUT_MD}")
    print(f"File TSV dibuat: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
