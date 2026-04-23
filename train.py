import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def train_model(data_path='dataset/landmark/asl_landmarks.csv'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Please run convert_image_to_landmarks.py first.")
        return

    print(f"Loading landmarks from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # ── C.1.8: 5-fold stratified cross-validation ────────────────────────────
    print("\nRunning 5-fold stratified cross-validation (this may take a minute)...")
    cv_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    cv_summary = (
        f"Cross-Validation Accuracy (5-fold): "
        f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n"
        f"Per-fold scores: {np.round(cv_scores, 4)}\n\n"
    )
    print(cv_summary)

    # ── Final train/test split for the held-out evaluation report ────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Random Forest Classifier on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 1. Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)
    
    with open('results/classification_report.txt', 'w') as f:
        f.write(cv_summary)
        f.write(report)
        
    # 2. Visual Evaluation: Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix: Sign Recognition')
    plt.ylabel('Actual Sign')
    plt.xlabel('Predicted Sign')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    print("Saved confusion_matrix.png to results/")

    # 3. Visual Evaluation: Feature Importance (Top 10)
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.title('Top 10 Important Features (Landmarks)')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [f'L_{i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    print("Saved feature_importance.png to results/")
    
    # Save the model
    joblib.dump(model, 'sign_classifier.pkl')
    print(f"\nModel saved as sign_classifier.pkl")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Hold-out test accuracy: {(y_pred == y_test).mean():.4f}")

if __name__ == "__main__":
    train_model()
