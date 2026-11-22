"""
COMP 365: Campus Parking Predictor — Practical Checkpoint 3
Team: Parallel Perfection
Members: Aniya Teasley, Brianna Williams, Daiyana Brooks, Natalia Smythe
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

SEED = 365

# ==============================================================================
# 1. Input Data (Demonstration Synthetic Dataset)
# ==============================================================================
input_file = Path('parking_observations.csv')
if input_file.exists():
    raw = pd.read_csv(input_file)
    data_source = 'TEAM CSV (user supplied; verify collection provenance)'
else:
    rng = np.random.default_rng(SEED)
    areas = ['Area A', 'Area B', 'Area C']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = ['08:00', '10:00', '12:00', '14:00', '16:00']
    rows = []
    for repeat in range(5):
        for area in areas:
            for day in days:
                for hour in hours:
                    busy = (hour in ['10:00', '12:00']) + (area == 'Area A') + (day in ['Tuesday', 'Thursday'])
                    p_open = {0: 0.87, 1: 0.68, 2: 0.35, 3: 0.16}[int(busy)]
                    rows.append([area, day, hour, int(rng.random() < p_open)])
    raw = pd.DataFrame(rows, columns=['parking_area', 'day_of_week', 'time_of_day', 'available'])
    data_source = 'SYNTHETIC DEMONSTRATION ONLY'
    raw.to_csv('synthetic_parking_demo.csv', index=False)

print(f"Data source: {data_source} | Total records: {len(raw)}")

# ==============================================================================
# 2. Data Cleaning & Validation (Aniya Teasley)
# ==============================================================================
df = raw.copy()
for col in ['parking_area', 'day_of_week', 'time_of_day']:
    df[col] = df[col].astype(str).str.strip()

df = df.dropna(subset=['parking_area', 'day_of_week', 'time_of_day', 'available'])
df['available'] = pd.to_numeric(df['available'], errors='coerce')
df = df.dropna(subset=['available'])
df['available'] = df['available'].astype(int)

# ==============================================================================
# 3. Model Suite: 3 ML Models vs. Majority-Class Baseline (Natalia Smythe)
# ==============================================================================
features = ['parking_area', 'day_of_week', 'time_of_day']
X, y = df[features], df['available']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

preprocessor = ColumnTransformer([
    ('categories', OneHotEncoder(handle_unknown='ignore'), features)
])

models = {
    'Majority Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(random_state=SEED),
    'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED),
    'Random Forest (n=50)': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=3, random_state=SEED)
}

pipelines = {}
results_records = []
predictions = {}

print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Held-out class distribution: Class 0 (No space) = {(y_test == 0).sum()} | Class 1 (Space open) = {(y_test == 1).sum()}")
print("=" * 80)

for name, clf in models.items():
    pipe = Pipeline([('encode', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe
    
    pred = pipe.predict(X_test)
    predictions[name] = pred
    
    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    
    results_records.append({
        'method': name,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    })

results = pd.DataFrame(results_records)
print("\n=== MULTI-MODEL BENCHMARK RESULTS ===")
print(results.to_string(index=False))

print("\n=== DECISION TREE CLASSIFICATION REPORT ===")
print(classification_report(y_test, predictions['Decision Tree (depth=5)'], target_names=['No space (0)', 'Space open (1)'], zero_division=0))

print("\n=== RANDOM FOREST CLASSIFICATION REPORT ===")
print(classification_report(y_test, predictions['Random Forest (n=50)'], target_names=['No space (0)', 'Space open (1)'], zero_division=0))

# ==============================================================================
# 4. Live Interactive Parking Availability Inference
# ==============================================================================
def predict_parking_availability(lot, day, time_slot, model_name='Random Forest (n=50)'):
    model = pipelines[model_name]
    sample = pd.DataFrame([{'parking_area': lot, 'day_of_week': day, 'time_of_day': time_slot}])
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    
    status = '🟢 Space Open (Available)' if pred == 1 else '🔴 Full (No Space Expected)'
    conf = proba[pred] * 100
    
    print(f"Query: Location: {lot:<8} | Day: {day:<9} | Time: {time_slot} | Model: {model_name}")
    print(f"Prediction: {status} (Confidence: {conf:.1f}%)\n")
    return pred

print("\n=== LIVE INFERENCE DEMONSTRATIONS ===")
predict_parking_availability('Area A', 'Tuesday', '10:00')  # High-traffic morning block
predict_parking_availability('Area B', 'Monday', '08:00')   # Early morning transition
predict_parking_availability('Area C', 'Friday', '16:00')   # Late afternoon
