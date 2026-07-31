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
    df[col] = df[col].replace(0, np.nan)

# Separate Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute Data
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train = pd.DataFrame(X_train_imputed, columns=X.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X.columns)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# --- 1. Train Baseline Model ---
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train_scaled, y_train)

# --- 2. Train GWO-Optimized Model ---
# Hardcode the best parameters found by GWO from PROGRESS.md
gwo_depth = 4
gwo_leaf = 21
gwo_model = DecisionTreeClassifier(
    max_depth=gwo_depth,
    min_samples_leaf=gwo_leaf,
    criterion='gini',
    random_state=42
)
gwo_model.fit(X_train_scaled, y_train)

# --- 3. Train RandomizedSearchCV Model ---
# Use parameters from PROGRESS.md
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(1, 21)),
    'min_samples_leaf': list(range(1, 51))
}
base_model_rs = DecisionTreeClassifier(random_state=42)
rand_search = RandomizedSearchCV(
    estimator=base_model_rs,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='recall',
    random_state=42,
    n_jobs=-1
)
rand_search.fit(X_train_scaled, y_train)
rand_model = rand_search.best_estimator_

# --- 4. Evaluate Models ---
y_pred_base = base_model.predict(X_test_scaled)
y_pred_gwo = gwo_model.predict(X_test_scaled)
y_pred_rand = rand_model.predict(X_test_scaled)

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
print("Tabel Perbandingan Metrik")
print(comparison_df)
