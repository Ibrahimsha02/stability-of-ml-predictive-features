import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from collections import Counter
 
# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
X, y = make_classification(
    n_samples=500, n_features=15, n_informative=5,
    n_redundant=2, n_repeated=0, n_classes=2, random_state=42
)
feature_names = [f"F{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. Define Jaccard similarity
# -----------------------------
def jaccard_index(list1, list2):
    set1, set2 = set(list1), set(list2)
    return len(set1 & set2) / len(set1 | set2)
 
# -----------------------------
# 3. Feature selection stability
# -----------------------------
n_iterations = 20     # Number of subsamples
sample_size = 50      # Limited data
top_k = 5             # Top features to select
 
selected_features_rf = []
selected_features_lr = []
 
for i in range(n_iterations):
    # Random subsample
    df_sample = df.sample(n=sample_size, random_state=i)
    X_sample = df_sample[feature_names].values
    y_sample = df_sample['target'].values
 
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
# --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=i)
    rf.fit(X_scaled, y_sample)
    importances = rf.feature_importances_
    top_features_rf = np.argsort(importances)[-top_k:]
    selected_features_rf.append([feature_names[j] for j in top_features_rf])
 
    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=i)
    lr.fit(X_scaled, y_sample)
    coefs = np.abs(lr.coef_[0])
    top_features_lr = np.argsort(coefs)[-top_k:]
    selected_features_lr.append([feature_names[j] for j in top_features_lr])
# -----------------------------
# 4. Compute Jaccard stability
# -----------------------------
stability_rf = [jaccard_index(f1, f2) for f1, f2 in combinations(selected_features_rf, 2)]
stability_lr = [jaccard_index(f1, f2) for f1, f2 in combinations(selected_features_lr, 2)]
print(f"Average Stability (Random Forest): {np.mean(stability_rf):.3f}")
print(f"Average Stability (Logistic Regression): {np.mean(stability_lr):.3f}")
# -----------------------------
# 5. Feature selection frequency plot
# -----------------------------
all_selected_rf = [f for sublist in selected_features_rf for f in sublist]
all_selected_lr = [f for sublist in selected_features_lr for f in sublist]
 
freq_rf = Counter(all_selected_rf)
freq_lr = Counter(all_selected_lr)
 
plt.figure(figsize=(12,5))
plt.bar(np.arange(len(freq_rf)), list(freq_rf.values()), tick_label=list(freq_rf.keys()), alpha=0.7, label="Random Forest")
plt.bar(np.arange(len(freq_lr)), list(freq_lr.values()), tick_label=list(freq_lr.keys()), alpha=0.5, label="Logistic Regression")
plt.xlabel("Features")
plt.ylabel("Selection Frequency")
plt.title("Feature Selection Frequency under Limited Data")
plt.legend()
plt.grid()
plt.show()