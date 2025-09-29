import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

df = pd.read_csv("train.csv")
df = df.drop(columns = ["Id"])

x = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]

numeric_features = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

categorical_features = [col for col in x.columns if col not in numeric_features]

scaler = StandardScaler()
x_scaled = x.copy()
x_scaled[numeric_features] = scaler.fit_transform(x[numeric_features])

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/cover_type_prediction_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅Model and scaler saved in 'models/' directory.")