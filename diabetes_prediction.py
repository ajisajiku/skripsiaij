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
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)


# --- Train GWO-Optimized Model (needed for SHAP) ---
# Hardcode the best parameters found by GWO
best_max_depth = 4
best_min_samples_leaf = 21

gwo_model = DecisionTreeClassifier(
    max_depth=best_max_depth,
    min_samples_leaf=best_min_samples_leaf,
    criterion='gini',
    random_state=42
)
gwo_model.fit(X_train, y_train)


# --- XAI with SHAP ---
print("\n--- Generating SHAP Explanations ---")

# Get feature names from X_train
feature_names = X_train.columns.tolist()

# Create TreeExplainer for the GWO model
explainer = shap.TreeExplainer(gwo_model)

# Calculate SHAP values for the training data
shap_values_train = explainer(X_train)

# Display global feature importance (SHAP summary plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_train, X_train, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Global)")
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.close()

print("SHAP summary plot saved as 'shap_summary_plot.png'.")