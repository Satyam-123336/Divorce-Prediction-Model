import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

df = pd.read_csv("divorce.csv", sep=";")
print("=== DATASET ===")
print(f"Total samples : {len(df)}")
counts = df["Class"].value_counts()
ratios = df["Class"].value_counts(normalize=True).round(3)
print(f"Stable  (0)   : {counts[0]}  ({ratios[0]*100:.1f}%)")
print(f"Divorce (1)   : {counts[1]}  ({ratios[1]*100:.1f}%)")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model = joblib.load("divorce_model.pkl")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print()
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Stable (0)", "Divorce (1)"]))

print("=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(f"                   Pred Stable   Pred Divorce")
print(f"Actual Stable           {cm[0][0]}              {cm[0][1]}")
print(f"Actual Divorce          {cm[1][0]}              {cm[1][1]}")

print()
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score : {auc:.4f}")

print()
print("=== CROSS-VALIDATION (5-fold stratified) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
print(f"F1 per fold   : {[round(s, 4) for s in scores]}")
print(f"Mean F1       : {scores.mean():.4f}   Std: {scores.std():.4f}")

print()
print("=== BIAS CHECK: EXTREME INPUTS ===")
all_zero = np.zeros((1, 54))
all_four = np.full((1, 54), 4)
all_two  = np.full((1, 54), 2)
p0 = model.predict_proba(all_zero)[0]
p4 = model.predict_proba(all_four)[0]
p2 = model.predict_proba(all_two)[0]
print(f"All 0 (Never)      -> Stable: {p0[0]*100:.1f}%  Divorce: {p0[1]*100:.1f}%  => {'Divorce' if p0[1]>0.5 else 'Stable'}")
print(f"All 2 (Sometimes)  -> Stable: {p2[0]*100:.1f}%  Divorce: {p2[1]*100:.1f}%  => {'Divorce' if p2[1]>0.5 else 'Stable'}")
print(f"All 4 (Always)     -> Stable: {p4[0]*100:.1f}%  Divorce: {p4[1]*100:.1f}%  => {'Divorce' if p4[1]>0.5 else 'Stable'}")

print()
print("=== TOP 10 MOST INFLUENTIAL FEATURES ===")
importances = model.feature_importances_
cols = list(X.columns)
top10 = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)[:10]
for feat, imp in top10:
    print(f"  {feat:8s}  {imp*100:.2f}%")
