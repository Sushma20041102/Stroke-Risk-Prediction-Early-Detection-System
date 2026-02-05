import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

print("Loading dataset...")

df = pd.read_csv("stroke_data.csv")

# Drop id column if present
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Remove missing rows
df = df.dropna()

# Columns to encode
categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# Save encoders so API can use the same mapping
encoders = {}

for col in categorical_cols:
    df[col] = df[col].astype("category")
    encoders[col] = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes

# Separate features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost...")

model = xgb.XGBClassifier(enable_categorical=True)
model.fit(X_train, y_train)

# Save everything needed for inference
joblib.dump(model, "stroke_xgb.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Training complete!")
print("Saved: stroke_xgb.pkl and encoders.pkl")
