from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib  # ✅ use joblib, not your test file

# Load your dataset
df = pd.read_csv("E:/text-predictor/data/english.csv")

# Train LabelEncoder
le = LabelEncoder()
le.fit(df['label'])  # make sure 'label' is the correct column name in your CSV

# Save encoder
joblib.dump(le, "model/label_encoder.pkl")  # ✅ save inside your models folder
