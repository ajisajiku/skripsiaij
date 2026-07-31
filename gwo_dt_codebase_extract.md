# GWO-DT Codebase Extract

Generated from local workspace `C:\xampp\htdocs\dm`. Values are recomputed from `diabetes.csv` using the same preprocessing pipeline as the application unless otherwise noted.

# 1. REPOSITORY STRUCTURE

## Complete File/Folder Inventory

| Type | Path | Size bytes |
| --- | --- | --- |
| Python | app.py | 13524 |
| Python | bab4_experiment_report.py | 18817 |
| Python | comparison.py | 3253 |
| Python | diabetes_prediction.py | 2252 |
| Python | gwo_optimizer.py | 3607 |
| Python | interactive_prediction.py | 2990 |
| Python | model_utils.py | 7238 |
| Data/Artifact | BAB3_METODOLOGI.md | 12118 |
| Data/Artifact | BAB4_HASIL_DAN_PEMBAHASAN.md | 19520 |
| Data/Artifact | BAB5_KESIMPULAN_SARAN.md | 3892 |
| Data/Artifact | BASELINE_MODEL_EXPLANATION.md | 6632 |
| Data/Artifact | EXPLANATION_NOTE.md | 2933 |
| Data/Artifact | GWO_EXPLANATION.md | 3982 |
| Data/Artifact | GWO_VS_RANDOMIZEDSEARCH_EXPLANATION.md | 4017 |
| Data/Artifact | PROGRESS.md | 6961 |
| Data/Artifact | RANDOMIZEDSEARCH_EXPLANATION.md | 3967 |
| Data/Artifact | bab4_class_distribution_verification.md | 2630 |
| Data/Artifact | bab4_class_distribution_verification.tsv | 650 |
| Data/Artifact | bab4_decisiontree_baseline_verification.md | 6637 |
| Data/Artifact | bab4_decisiontree_baseline_verification.tsv | 3820 |
| Data/Artifact | bab4_experiment_report_output.md | 10935 |
| Data/Artifact | bab4_experiment_tables.tsv | 7632 |
| Data/Artifact | bab4_gwo_final_model_verification.md | 3454 |
| Data/Artifact | bab4_gwo_final_model_verification.tsv | 1567 |
| Data/Artifact | bab4_gwo_fitness_history.tsv | 712 |
| Data/Artifact | bab4_gwo_repeatability_check.md | 2631 |
| Data/Artifact | bab4_gwo_verification.md | 7120 |
| Data/Artifact | bab4_gwo_verification.tsv | 3192 |
| Data/Artifact | bab4_imputation_verification.md | 3592 |
| Data/Artifact | bab4_imputation_verification.tsv | 1763 |
| Data/Artifact | bab4_median_verification.md | 2473 |
| Data/Artifact | bab4_median_verification.tsv | 434 |
| Data/Artifact | bab4_pipeline_audit.md | 9563 |
| Data/Artifact | bab4_randomizedsearch_verification.md | 6457 |
| Data/Artifact | bab4_randomizedsearch_verification.tsv | 2618 |
| Data/Artifact | bab4_standardscaler_verification.md | 5724 |
| Data/Artifact | bab4_standardscaler_verification.tsv | 2569 |
| Data/Artifact | confusion_matrix.png | 18965 |
| Data/Artifact | data_distribution.png | 30162 |
| Data/Artifact | diabetes.csv | 23873 |
| Data/Artifact | diagram1.xml | 8113 |
| Data/Artifact | diagram2.xml | 7552 |
| Data/Artifact | diagram3.xml | 2817 |
| Data/Artifact | final_confusion_matrix.png | 16429 |
| Data/Artifact | gwo_model.joblib | 3641 |
| Data/Artifact | hitungmanual.md | 35780 |
| Data/Artifact | perbandingan metrik.png | 24371 |
| Data/Artifact | roc_curve.png | 41156 |
| Data/Artifact | scaler.joblib | 1159 |
| Data/Artifact | shap_summary_plot.png | 52460 |

## Directory Tree (2-3 Levels Deep)

```text
dm/
+-- __pycache__/
|   +-- gwo_optimizer.cpython-313.pyc
|   +-- interactive_prediction.cpython-313.pyc
|   +-- model_utils.cpython-313.pyc
+-- app.py
+-- BAB3_METODOLOGI.md
+-- bab4_class_distribution_verification.md
+-- bab4_class_distribution_verification.tsv
+-- bab4_decisiontree_baseline_verification.md
+-- bab4_decisiontree_baseline_verification.tsv
+-- bab4_experiment_report.py
+-- bab4_experiment_report_output.md
+-- bab4_experiment_tables.tsv
+-- bab4_gwo_final_model_verification.md
+-- bab4_gwo_final_model_verification.tsv
+-- bab4_gwo_fitness_history.tsv
+-- bab4_gwo_repeatability_check.md
+-- bab4_gwo_verification.md
+-- bab4_gwo_verification.tsv
+-- BAB4_HASIL_DAN_PEMBAHASAN.md
+-- bab4_imputation_verification.md
+-- bab4_imputation_verification.tsv
+-- bab4_median_verification.md
+-- bab4_median_verification.tsv
+-- bab4_pipeline_audit.md
+-- bab4_randomizedsearch_verification.md
+-- bab4_randomizedsearch_verification.tsv
+-- bab4_standardscaler_verification.md
+-- bab4_standardscaler_verification.tsv
+-- BAB5_KESIMPULAN_SARAN.md
+-- BASELINE_MODEL_EXPLANATION.md
+-- comparison.py
+-- confusion_matrix.png
+-- data_distribution.png
+-- diabetes.csv
+-- diabetes_prediction.py
+-- diagram1.xml
+-- diagram2.xml
+-- diagram3.xml
+-- EXPLANATION_NOTE.md
+-- final_confusion_matrix.png
+-- GWO_EXPLANATION.md
+-- gwo_model.joblib
+-- gwo_optimizer.py
+-- GWO_VS_RANDOMIZEDSEARCH_EXPLANATION.md
+-- hitungmanual.md
+-- interactive_prediction.py
+-- model_utils.py
+-- perbandingan metrik.png
+-- PROGRESS.md
+-- RANDOMIZEDSEARCH_EXPLANATION.md
+-- roc_curve.png
+-- scaler.joblib
+-- shap_summary_plot.png
```

## Main Python Files: First 10-15 Lines and Purpose

### `app.py`

Purpose: Streamlit application entry point: page setup, module navigation, preprocessing trigger, model training tabs, evaluation, and interactive prediction.

```python
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
```

### `bab4_experiment_report.py`

Purpose: Reproducible report generator for Bab 4 experiment tables and markdown/TSV outputs.

```python
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
```

### `comparison.py`

Purpose: Standalone script that rebuilds the preprocessing pipeline, trains baseline/GWO/RandomizedSearch models, and prints metric comparison.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Load Data
df = pd.read_csv('diabetes.csv')

# Replace 0s with NaN
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_replace:
```

### `diabetes_prediction.py`

Purpose: Standalone script that trains a hard-coded GWO Decision Tree and generates a SHAP feature-importance plot.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import shap
import matplotlib.pyplot as plt

# --- STAGE 1: DATA PREPROCESSING ---

# Load Data
df = pd.read_csv('diabetes.csv')

# Replace 0s with NaN
```

### `gwo_optimizer.py`

Purpose: Grey Wolf Optimizer implementation and Decision Tree fitness function based on 5-fold recall.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import random
import streamlit as st

def calculate_fitness(params, X_train_scaled, y_train):
    max_depth = int(params[0])
    min_samples_leaf = int(params[1])

    if max_depth < 1 or min_samples_leaf < 1:
        return 1.0

    model = DecisionTreeClassifier(
        max_depth=max_depth,
```

### `interactive_prediction.py`

Purpose: Streamlit form module for manual patient input, preprocessing, prediction, probability display, and SHAP explanation.

```python
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
```

### `model_utils.py`

Purpose: Reusable model utilities for preprocessing, baseline training, RandomizedSearchCV training, and Streamlit evaluation views.

```python
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
```

# 2. PREPROCESSING PIPELINE

## Data Loading

Actual dataset path: `diabetes.csv` (`C:\xampp\htdocs\dm\diabetes.csv`).

Dataset shape: `768 rows x 9 columns`.

Column names: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.

| Column | Dtype |
| --- | --- |
| Pregnancies | int64 |
| Glucose | int64 |
| BloodPressure | int64 |
| SkinThickness | int64 |
| Insulin | int64 |
| BMI | float64 |
| DiabetesPedigreeFunction | float64 |
| Age | int64 |
| Outcome | int64 |

Code implementation:

```python
df = pd.read_csv('diabetes.csv')
```

## Missing Values Handling

Method: the application first treats physiologically invalid `0` values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing (`NaN`), then imputes with `SimpleImputer(strategy="median")` fitted on the training set only.

Code implementation:

```python
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_replace:
    df_input[col] = df_input[col].replace(0, np.nan)

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

Missing values per column after zero replacement:

| Column | Missing count after zero->NaN |
| --- | --- |
| Pregnancies | 0 |
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |
| DiabetesPedigreeFunction | 0 |
| Age | 0 |
| Outcome | 0 |

Training medians used for imputation:

| Feature | Median from training set |
| --- | --- |
| Pregnancies | 3.000000 |
| Glucose | 118.000000 |
| BloodPressure | 72.000000 |
| SkinThickness | 28.500000 |
| Insulin | 120.000000 |
| BMI | 32.000000 |
| DiabetesPedigreeFunction | 0.372500 |
| Age | 29.000000 |

## Normalization

Method: `StandardScaler()`. Parameters: default scikit-learn settings, `with_mean=True`, `with_std=True`; scaler is fitted on imputed training features only.

Features normalized: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`.

Code implementation:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Before/after examples from first training row:

| Feature | Original imputed value | Mean | Std used by StandardScaler | Formula | Scaled value |
| --- | --- | --- | --- | --- | --- |
| Glucose | 84.000000 | 121.815961 | 30.079488 | (84.0000 - 121.8160) / 30.0795 | -1.257201 |
| BMI | 32.000000 | 32.348208 | 6.929968 | (32.0000 - 32.3482) / 6.9300 | -0.050247 |

## Train/Test Split

Ratio: 80/20 (`test_size=0.2`). Random state: `42`. Stratification: not used (`stratify=None`).

Code:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Resulting shapes:

| Subset | Shape |
| --- | --- |
| X_train | (614, 8) |
| X_test | (154, 8) |
| y_train | (614,) |
| y_test | (154,) |

## Dataset Statistics

Original dataset numeric statistics:

| Feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| Pregnancies | 3.845052 | 3.369578 | 0.000000 | 3.000000 | 17.000000 |
| Glucose | 120.894531 | 31.972618 | 0.000000 | 117.000000 | 199.000000 |
| BloodPressure | 69.105469 | 19.355807 | 0.000000 | 72.000000 | 122.000000 |
| SkinThickness | 20.536458 | 15.952218 | 0.000000 | 23.000000 | 99.000000 |
| Insulin | 79.799479 | 115.244002 | 0.000000 | 30.500000 | 846.000000 |
| BMI | 31.992578 | 7.884160 | 0.000000 | 32.000000 | 67.100000 |
| DiabetesPedigreeFunction | 0.471876 | 0.331329 | 0.078000 | 0.372500 | 2.420000 |
| Age | 33.240885 | 11.760232 | 21.000000 | 29.000000 | 81.000000 |
| Outcome | 0.348958 | 0.476951 | 0.000000 | 0.000000 | 1.000000 |

Training data statistics after imputation and StandardScaler:

| Feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| Pregnancies | 0.000000 | 1.000815 | -1.130523 | -0.224334 | 4.004552 |
| Glucose | -0.000000 | 1.000815 | -2.587011 | -0.126863 | 2.566002 |
| BloodPressure | -0.000000 | 1.000815 | -3.989409 | -0.018995 | 4.116852 |
| SkinThickness | -0.000000 | 1.000815 | -2.447693 | -0.008140 | 4.097449 |
| Insulin | 0.000000 | 1.000815 | -1.401201 | -0.204516 | 7.991644 |
| BMI | 0.000000 | 1.000815 | -2.041598 | -0.050247 | 5.014712 |
| DiabetesPedigreeFunction | 0.000000 | 1.000815 | -1.162210 | -0.287212 | 5.796175 |
| Age | -0.000000 | 1.000815 | -1.035940 | -0.339929 | 4.184145 |

Class distribution:

| Subset | Negative/0 | Positive/1 | Total |
| --- | --- | --- | --- |
| Full dataset | 500 | 268 | 768 |
| Training | 401 | 213 | 614 |
| Testing | 99 | 55 | 154 |

# 3. BASELINE DECISION TREE MODEL

## Model Initialization

All hyperparameters from `DecisionTreeClassifier(random_state=42)` after scikit-learn defaults:

| Parameter | Value |
| --- | --- |
| ccp_alpha | 0.000000 |
| class_weight |  |
| criterion | gini |
| max_depth |  |
| max_features |  |
| max_leaf_nodes |  |
| min_impurity_decrease | 0.000000 |
| min_samples_leaf | 1 |
| min_samples_split | 2 |
| min_weight_fraction_leaf | 0.000000 |
| monotonic_cst |  |
| random_state | 42 |
| splitter | best |

Complete initialization code:

```python
base_model = DecisionTreeClassifier(random_state=42)
```

## Training Process

Training code:

```python
base_model.fit(X_train_scaled, y_train)
cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=5, scoring='recall')
```

Measured local training time during extraction: `0.008279` seconds. Validation used in app tab: 5-fold cross-validation recall via `cross_val_score`.

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline DT | 0.727273 | 0.610169 | 0.654545 | 0.631579 | 0.711111 | 36 | 23 | 19 | 76 |

# 4. GREY WOLF OPTIMIZER (GWO) IMPLEMENTATION

## GWO Algorithm Code

Complete `gwo_optimizer.py` content:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import random
import streamlit as st

def calculate_fitness(params, X_train_scaled, y_train):
    max_depth = int(params[0])
    min_samples_leaf = int(params[1])

    if max_depth < 1 or min_samples_leaf < 1:
        return 1.0

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion='gini',
        random_state=42
    )

    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall')
    avg_recall = np.mean(scores)

    return 1.0 - avg_recall

def run_gwo(fitness_function, lower_bound, upper_bound, dimensions, num_wolves, max_iterations, X_train_scaled, y_train):
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    alpha_pos = np.zeros(dimensions)
    alpha_score = float('inf')
    beta_pos = np.zeros(dimensions)
    beta_score = float('inf')
    delta_pos = np.zeros(dimensions)
    delta_score = float('inf')

    positions = np.zeros((num_wolves, dimensions))
    for i in range(num_wolves):
        positions[i, :] = np.random.uniform(0, 1, dimensions) * (upper_bound - lower_bound) + lower_bound

    progress_bar = st.progress(0)
    status_text = st.empty()

    for t in range(max_iterations):
        for i in range(num_wolves):
            positions[i,:] = np.clip(positions[i,:], lower_bound, upper_bound)
            fitness = fitness_function(positions[i, :], X_train_scaled, y_train)

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness > alpha_score and fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness > alpha_score and fitness > beta_score and fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        yield {
           'iteration': t + 1,
           'alpha_params': alpha_pos,
           'alpha_score': alpha_score,
           'beta_params': beta_pos,
           'beta_score': beta_score,
           'delta_params': delta_pos,
           'delta_score': delta_score
        }

        a = 2 - t * (2 / max_iterations)

        for i in range(num_wolves):
            for j in range(dimensions):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3
        
        progress_bar.progress((t + 1) / max_iterations)
        status_text.text(f'Iterasi {t+1}/{max_iterations}, Best Score (1 - Recall): {alpha_score:.4f}')
```

## GWO Components

Wolves are initialized with uniform random positions inside lower/upper bounds. Alpha, beta, and delta are the three best solutions tracked by lowest fitness. Fitness is `1 - mean(recall)` from 5-fold CV, so smaller is better. Position updates use the standard GWO equations with `A`, `C`, distance from alpha/beta/delta, and average of `X1`, `X2`, `X3`. Convergence criteria in the app is fixed iteration count; there is no early stopping.

## GWO Parameters

Application call parameters: population size `10`, maximum iterations `20`, dimensions `2`, lower bound `[1, 1]`, upper bound `[20, 50]`. Tuned parameters are `max_depth` and `min_samples_leaf`; `criterion` is fixed to `gini`, `random_state=42` for the Decision Tree. The app does not seed NumPy/Python random for GWO, so optimizer results can vary between runs. The saved final model uses `max_depth=4`, `min_samples_leaf=22`.

## Integration into Training

Actual integration from `app.py`:

```python
LOWER_BOUND, UPPER_BOUND, DIMENSIONS, NUM_WOLVES, MAX_ITERATIONS = [1, 1], [20, 50], 2, 10, 20

for update in run_gwo(calculate_fitness, LOWER_BOUND, UPPER_BOUND, DIMENSIONS, NUM_WOLVES, MAX_ITERATIONS, X_train_scaled, y_train):
    gwo_history.append(update['alpha_score'])
    last_gwo_update = update

best_params = last_gwo_update['alpha_params']
gwo_depth = int(best_params[0])
gwo_leaf = int(best_params[1])
gwo_model = DecisionTreeClassifier(max_depth=gwo_depth, min_samples_leaf=gwo_leaf, criterion='gini', random_state=42)
gwo_model.fit(X_train_scaled, y_train)
```

# 5. HYPERPARAMETER TUNING & OPTIMIZATION

## Search Space Definition

| Method | Hyperparameter | Min/Values | Max | Reason |
| --- | --- | --- | --- | --- |
| RandomizedSearchCV | criterion | gini, entropy | n/a | Compare impurity criteria supported by DecisionTreeClassifier. |
| RandomizedSearchCV/GWO | max_depth | 1 | 20 | Controls tree complexity; range covers shallow to moderately deep trees for this 768-row dataset. |
| RandomizedSearchCV/GWO | min_samples_leaf | 1 | 50 | Controls leaf size and overfitting; up to 50 forces smoother trees. |

## RandomizedSearch Implementation

Complete function from `model_utils.py`:

```python
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
```

Number of iterations/combinations tried: `100` sampled from `2000` possible combinations. Scoring metric: `recall`. Best parameters and score:

| Item | Value |
| --- | --- |
| best_params_ | {'min_samples_leaf': 30, 'max_depth': 4, 'criterion': 'entropy'} |
| best_score_ recall CV | 0.723367 |
| Measured extraction search time seconds | 4.618985 |

Top RandomizedSearch results:

| Parameters | Mean recall CV | Std recall CV | Rank |
| --- | --- | --- | --- |
| {'min_samples_leaf': 30, 'max_depth': 4, 'criterion': 'entropy'} | 0.723367 | 0.052559 | 1 |
| {'min_samples_leaf': 33, 'max_depth': 4, 'criterion': 'entropy'} | 0.723367 | 0.052559 | 1 |
| {'min_samples_leaf': 19, 'max_depth': 12, 'criterion': 'entropy'} | 0.670986 | 0.055615 | 3 |
| {'min_samples_leaf': 2, 'max_depth': 6, 'criterion': 'gini'} | 0.667331 | 0.083148 | 4 |
| {'min_samples_leaf': 40, 'max_depth': 6, 'criterion': 'entropy'} | 0.666999 | 0.097278 | 5 |
| {'min_samples_leaf': 32, 'max_depth': 6, 'criterion': 'gini'} | 0.657807 | 0.072946 | 6 |
| {'min_samples_leaf': 41, 'max_depth': 6, 'criterion': 'entropy'} | 0.657697 | 0.093839 | 7 |
| {'min_samples_leaf': 11, 'max_depth': 5, 'criterion': 'gini'} | 0.653267 | 0.080089 | 8 |
| {'min_samples_leaf': 35, 'max_depth': 12, 'criterion': 'gini'} | 0.653156 | 0.073956 | 9 |
| {'min_samples_leaf': 34, 'max_depth': 8, 'criterion': 'gini'} | 0.653156 | 0.073956 | 9 |

## GWO Optimization Implementation

Complete optimization loop is in Section 4. Fitness function: `1 - mean recall 5-fold CV`. Recorded convergence history from `bab4_gwo_fitness_history.tsv` if present:

| Iterasi | Best Fitness | Best Recall |
| --- | --- | --- |
| 1.000000 | 0.370764 | 0.629236 |
| 2.000000 | 0.342082 | 0.657918 |
| 3.000000 | 0.328793 | 0.671207 |
| 4.000000 | 0.328793 | 0.671207 |
| 5.000000 | 0.313621 | 0.686379 |
| 6.000000 | 0.295792 | 0.704208 |
| 7.000000 | 0.295792 | 0.704208 |
| 8.000000 | 0.295792 | 0.704208 |
| 9.000000 | 0.295792 | 0.704208 |
| 10.000000 | 0.253821 | 0.746179 |
| 11.000000 | 0.253821 | 0.746179 |
| 12.000000 | 0.253821 | 0.746179 |
| 13.000000 | 0.253821 | 0.746179 |
| 14.000000 | 0.253821 | 0.746179 |
| 15.000000 | 0.253821 | 0.746179 |
| 16.000000 | 0.253821 | 0.746179 |
| 17.000000 | 0.253821 | 0.746179 |
| 18.000000 | 0.253821 | 0.746179 |
| 19.000000 | 0.253821 | 0.746179 |
| 20.000000 | 0.253821 | 0.746179 |

Best saved final GWO model parameters:

| Parameter | Value |
| --- | --- |
| criterion | gini |
| max_depth | 4 |
| min_samples_leaf | 22 |
| min_samples_split | 2 |
| random_state | 42 |
| splitter | best |

## Comparison: RandomizedSearch vs GWO

RandomizedSearch best CV recall: `0.723367`. Recorded GWO best recall in `bab4_gwo_fitness_history.tsv`: `0.746179`. On the held-out test set, the saved final GWO model has higher recall than RandomizedSearch but lower accuracy/precision; RandomizedSearch has the highest F1 among the three in this extraction.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomizedSearch DT | 0.733766 | 0.612903 | 0.690909 | 0.649573 | 0.812029 | 38 | 24 | 17 | 75 |
| GWO DT (saved final) | 0.701299 | 0.560000 | 0.763636 | 0.646154 | 0.802847 | 42 | 33 | 13 | 66 |

# 6. DECISION TREE TRAINING WITH OPTIMIZATION

## Optimized Models

RandomizedSearch optimized Decision Tree code and result:

```python
rand_search.fit(X_train_scaled, y_train)
rand_model = rand_search.best_estimator_
y_pred_rand = rand_model.predict(X_test_scaled)
```

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomizedSearch DT | 0.733766 | 0.612903 | 0.690909 | 0.649573 | 0.812029 | 38 | 24 | 17 | 75 |

GWO optimized Decision Tree code and result (`gwo_model.joblib (persisted final model)`):

```python
gwo_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=22, criterion='gini', random_state=42)
gwo_model.fit(X_train_scaled, y_train)
y_pred_gwo = gwo_model.predict(X_test_scaled)
```

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GWO DT (saved final) | 0.701299 | 0.560000 | 0.763636 | 0.646154 | 0.802847 | 42 | 33 | 13 | 66 |

# 7. MODEL EVALUATION & TESTING

## Evaluation Methodology

Test set approach: train/test split with 154 held-out rows. Metrics calculated: accuracy, precision, recall, F1, AUC-ROC, and confusion matrix. Cross-validation strategy used for training/tuning: 5-fold CV on training set; app uses default non-stratified `cv=5` for `cross_val_score`/`RandomizedSearchCV` classifier scoring.

## Cross-Validation Results (if applicable)

| Model | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline DT | accuracy | 0.682927 | 0.747967 | 0.642276 | 0.666667 | 0.631148 | 0.674197 | 0.041103 |
| Baseline DT | precision | 0.529412 | 0.676471 | 0.489796 | 0.523810 | 0.459459 | 0.535789 | 0.074722 |
| Baseline DT | recall | 0.642857 | 0.534884 | 0.558140 | 0.511628 | 0.404762 | 0.530454 | 0.076907 |
| Baseline DT | f1 | 0.580645 | 0.597403 | 0.521739 | 0.517647 | 0.430380 | 0.529563 | 0.058740 |
| Baseline DT | roc_auc | 0.673280 | 0.698692 | 0.622820 | 0.630814 | 0.577381 | 0.640597 | 0.042074 |
| RandomizedSearch DT | accuracy | 0.731707 | 0.780488 | 0.764228 | 0.739837 | 0.811475 | 0.765547 | 0.028777 |
| RandomizedSearch DT | precision | 0.588235 | 0.700000 | 0.640000 | 0.612245 | 0.693878 | 0.646872 | 0.044083 |
| RandomizedSearch DT | recall | 0.714286 | 0.651163 | 0.744186 | 0.697674 | 0.809524 | 0.723367 | 0.052559 |
| RandomizedSearch DT | f1 | 0.645161 | 0.674699 | 0.688172 | 0.652174 | 0.747253 | 0.681492 | 0.036317 |
| RandomizedSearch DT | roc_auc | 0.789389 | 0.851453 | 0.848837 | 0.785320 | 0.836161 | 0.822232 | 0.028972 |
| GWO DT (saved final) | accuracy | 0.731707 | 0.821138 | 0.747967 | 0.731707 | 0.778689 | 0.762242 | 0.034082 |
| GWO DT (saved final) | precision | 0.595745 | 0.698113 | 0.620000 | 0.596154 | 0.653061 | 0.632615 | 0.038874 |
| GWO DT (saved final) | recall | 0.666667 | 0.860465 | 0.720930 | 0.720930 | 0.761905 | 0.746179 | 0.064662 |
| GWO DT (saved final) | f1 | 0.629213 | 0.770833 | 0.666667 | 0.652632 | 0.703297 | 0.684528 | 0.049390 |
| GWO DT (saved final) | roc_auc | 0.767931 | 0.888517 | 0.817878 | 0.798110 | 0.828274 | 0.820142 | 0.039895 |

## Test Set Results for All 3 Models

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| Baseline DT | 0.727273 | 0.610169 | 0.654545 | 0.631579 | 0.711111 |
| RandomizedSearch DT | 0.733766 | 0.612903 | 0.690909 | 0.649573 | 0.812029 |
| GWO DT (saved final) | 0.701299 | 0.560000 | 0.763636 | 0.646154 | 0.802847 |

## Confusion Matrix Details (all 3 models)

| Model | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| Baseline DT | 36 | 23 | 19 | 76 |
| RandomizedSearch DT | 38 | 24 | 17 | 75 |
| GWO DT (saved final) | 42 | 33 | 13 | 66 |

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline DT | 0.727273 | 0.610169 | 0.654545 | 0.631579 | 36 | 23 | 19 | 76 |
| RandomizedSearch DT | 0.733766 | 0.612903 | 0.690909 | 0.649573 | 38 | 24 | 17 | 75 |
| GWO DT (saved final) | 0.701299 | 0.560000 | 0.763636 | 0.646154 | 42 | 33 | 13 | 66 |

# 8. MANUAL PREDICTION EXAMPLE (STEP-BY-STEP)

## Sample Input Data

One actual example from the test set: dataset index `668`.

| Dataset index | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Actual Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 668.000000 | 6.000000 | 98.000000 | 58.000000 | 33.000000 | 190.000000 | 34.000000 | 0.430000 | 43.000000 | 0.000000 |

## Preprocessing Applied to Example

Formula used by `StandardScaler`: `z = (value - mean) / std`, where `mean` and `std` are fitted from the imputed training set.

| Feature | Original | After zero->NaN | After median imputation | Mean | Std | Scaled | Formula |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pregnancies | 6.000000 | 6.000000 | 6.000000 | 3.742671 | 3.310565 | 0.681856 | (6.0000 - 3.7427) / 3.3106 |
| Glucose | 98.000000 | 98.000000 | 98.000000 | 121.815961 | 30.079488 | -0.791768 | (98.0000 - 121.8160) / 30.0795 |
| BloodPressure | 58.000000 | 58.000000 | 58.000000 | 72.229642 | 12.089421 | -1.177033 | (58.0000 - 72.2296) / 12.0894 |
| SkinThickness | 33.000000 | 33.000000 | 33.000000 | 28.568404 | 8.403178 | 0.527371 | (33.0000 - 28.5684) / 8.4032 |
| Insulin | 190.000000 | 190.000000 | 190.000000 | 138.115635 | 88.578061 | 0.585747 | (190.0000 - 138.1156) / 88.5781 |
| BMI | 34.000000 | 34.000000 | 34.000000 | 32.348208 | 6.929968 | 0.238355 | (34.0000 - 32.3482) / 6.9300 |
| DiabetesPedigreeFunction | 0.430000 | 0.430000 | 0.430000 | 0.469168 | 0.336572 | -0.116372 | (0.4300 - 0.4692) / 0.3366 |
| Age | 43.000000 | 43.000000 | 43.000000 | 32.907166 | 11.494065 | 0.878091 | (43.0000 - 32.9072) / 11.4941 |

## Decision Tree Path Trace (GWO-optimized model)

Prediction with saved GWO model: class `1` (`Diabetes`). Probabilities `[class 0, class 1]`: `[0.46729 0.53271]`.

| Node | Feature | Scaled value | Threshold | Rule | Direction | Next node |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Glucose | -0.791768 | 0.188967 | Glucose <= 0.188967 | LEFT | 1 |
| 1 | Age | 0.878091 | -0.383430 | Age <= -0.383430 | RIGHT | 9 |
| 9 | BMI | 0.238355 | -0.865546 | BMI <= -0.865546 | RIGHT | 11 |
| 11 | Glucose | -0.791768 | -0.908126 | Glucose <= -0.908126 | RIGHT | 13 |
| 13 | LEAF |  |  | class counts/proportions=[0.4672897196261682, 0.5327102803738317] | predict 1 |  |

## Path Sequence

Node 0 LEFT -> Node 1 RIGHT -> Node 9 RIGHT -> Node 11 RIGHT -> Node 13 predict 1
