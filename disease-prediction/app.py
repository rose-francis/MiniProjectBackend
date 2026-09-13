# app.py AKA Training Pipeline for Disease Prediction
# Trains (or loads, if already trained) the Random Forest disease-prediction model.
# Run with: python app.py
# Produces model.pkl / encoder.pkl / symptoms.pkl, loaded by donor-matching/api.py.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import os

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"
SYMPTOMS_PATH = "symptoms.pkl"


def train_and_save():
    CSV_PATH = "fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv"
    df = pd.read_csv(CSV_PATH)
    TARGET_COL = "prognosis"

    SYMPTOM_COLS = [c for c in df.columns if c != TARGET_COL]

    # Augmentation for rare classes
    MIN_SAMPLES = 20
    augmented_rows = []
    for disease, count in df[TARGET_COL].value_counts().items():
        if count < MIN_SAMPLES:
            disease_rows = df[df[TARGET_COL] == disease]
            needed = MIN_SAMPLES - count
            for _ in range(needed):
                row = disease_rows.sample(1, replace=True).iloc[0].copy()
                flip_mask = np.random.random(len(SYMPTOM_COLS)) < 0.05
                for col, flip in zip(SYMPTOM_COLS, flip_mask):
                    if flip:
                        row[col] = 1 - row[col]
                augmented_rows.append(row)
    if augmented_rows:
        df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_raw = df[TARGET_COL].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    symptom_names = [c for c in df.columns if c != TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(encoder, open(ENCODER_PATH, "wb"))
    pickle.dump(symptom_names, open(SYMPTOMS_PATH, "wb"))
    print("Model trained and saved.")
    return model, encoder, symptom_names


if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
    disease_encoder = pickle.load(open(ENCODER_PATH, "rb"))
    SYMPTOM_NAMES = pickle.load(open(SYMPTOMS_PATH, "rb"))
    print("Loaded model from disk.")
else:
    model, disease_encoder, SYMPTOM_NAMES = train_and_save()
