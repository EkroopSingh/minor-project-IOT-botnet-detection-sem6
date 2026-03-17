# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# =========================
# LOAD DATA
# =========================
print("Loading data...")

benign = pd.read_csv("data/raw/Danmini_Doorbell/benign_traffic.csv", nrows=30000)
attack = pd.read_csv("data/raw/Danmini_Doorbell/mirai_udp.csv", nrows=30000)

# Labels
benign["label"] = 0
attack["label"] = 1

# Combine
df = pd.concat([benign, attack])

print("Data Loaded:", df.shape)

# =========================
# CLEAN DATA
# =========================
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("After Cleaning:", df.shape)

# =========================
# SPLIT DATA
# =========================
X = df.drop("label", axis=1)
y = df["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
print("Training model...")

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

print("Model Trained ✅")

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# RESULTS
# =========================
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "outputs/models/iot_model.pkl")
print("\nModel Saved in outputs/models/ ✅")


import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CLASS DISTRIBUTION
# =========================
df['label'].value_counts().plot(kind='bar')
plt.title("Class Distribution (0=Normal, 1=Attack)")
plt.savefig("outputs/class_distribution.png")
plt.show()

# =========================
# CONFUSION MATRIX GRAPH
# =========================
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
import pandas as pd

importance = model.feature_importances_
features = pd.Series(importance, index=X.columns)

features.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.savefig("outputs/feature_importance.png")
plt.show()
