import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib 
import os

# STEP 1: Load dataset

df = pd.read_csv("diabetes_prediction_dataset.csv")
print("Dataset loaded!\n")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

#exit()

# STEP 2: Encode categorical column

le = LabelEncoder()
df["smoking_history"] = le.fit_transform(df["smoking_history"])

# STEP 3: Identify features (X) and target (y)

X = df.drop(["diabetes", "gender"], axis=1)   # gender removed
y = df["diabetes"]

print("\nFinal training features:", X.columns.tolist())
print("Target shape:", y.shape)

# STEP 4: Split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# STEP 5: Train the model

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("\nModel training completed.")
print("Model expects features:", model.feature_names_in_)

# STEP 6: Check accuracy

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Base accuracy
majority_class = df["diabetes"].mode()[0]
base_accuracy = sum(df["diabetes"] == majority_class) / len(df)
print("Base Accuracy:", base_accuracy)

# STEP 7: Predict for new data

sample_df = pd.read_csv("sample.csv")
names = sample_df["name"]

sample_inputs = sample_df.drop("name", axis=1)
sample_inputs["smoking_history"] = le.transform(sample_inputs["smoking_history"])

sample_predictions = model.predict(sample_inputs)

print("\nPredictions for sample.csv:")
for name, pred in zip(names, sample_predictions):
    status = "Diabetes" if pred == 1 else "No Diabetes"
    print(f"{name}: {status}")



# STEP 11: Save model + encoder

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel and Label Encoder saved successfully!")
