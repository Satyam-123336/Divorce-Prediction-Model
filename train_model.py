import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("divorce.csv", sep=";")
x = df.drop("Class", axis=1)
y = df["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)
print(f"Train Acc: {clf.score(x_train, y_train):.4f}")
print(f"Test  Acc: {clf.score(x_test, y_test):.4f}")
joblib.dump(clf, "divorce_model.pkl")
print("Model saved to divorce_model.pkl")
