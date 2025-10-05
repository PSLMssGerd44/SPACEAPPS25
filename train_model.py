import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import the ensemble and pipeline tools
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Import the three models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def train_ensemble_model():
    """
    Trains the ensemble model and SAVES performance metrics for the Streamlit app.
    """
    print("Starting ensemble model training on the merged dataset...")

    try:
        df = pd.read_csv('exoplanet_data_merged_for_ensemble.csv')
    except FileNotFoundError:
        print("Error: 'exoplanet_data_merged_for_ensemble.csv' not found.")
        print("Please run 'data_preparation.py' first.")
        return

    X = df.drop('disposition', axis=1)
    y = df['disposition']
    feature_columns = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- Create the Ensemble Model Pipeline ---
    clf1 = LGBMClassifier(random_state=42)
    clf2 = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    clf3 = CatBoostClassifier(random_state=42, verbose=0, loss_function='MultiClass')

    eclf1 = VotingClassifier(
        estimators=[('lgbm', clf1), ('xgb', clf2), ('catboost', clf3)],
        voting='soft'
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('ensemble', eclf1)
    ])

    print("Training the full pipeline (Scaler -> SMOTE -> Ensemble)...")
    pipeline.fit(X_train, y_train)

    # --- Evaluate the Model and Prepare Metrics for Saving ---
    print("\nEvaluating model performance on the test set...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['False Positive', 'Candidate', 'Confirmed'], output_dict=True)
    
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report (Text):")
    print(classification_report(y_test, y_pred, target_names=['False Positive', 'Candidate', 'Confirmed']))

    # --- NEW: Get Feature Importances ---
    # We access the trained models inside the pipeline and average their feature importances
    base_models = pipeline.named_steps['ensemble'].estimators_
    importances = [model.feature_importances_ for model in base_models]
    avg_importances = np.mean(importances, axis=0)
    
    feature_importance_data = pd.DataFrame({'feature': feature_columns, 'importance': avg_importances})
    feature_importance_data = feature_importance_data.sort_values(by='importance', ascending=False)
    
    # --- NEW: Save all metrics in a dictionary ---
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'y_test': y_test.to_numpy(),
        'y_pred': y_pred,
        'feature_importances': feature_importance_data
    }
    
    # --- Save all Assets ---
    joblib.dump(pipeline, 'exoplanet_ensemble_model.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    joblib.dump(metrics, 'training_metrics.joblib') # Save the new metrics file
    
    print("\nEnsemble pipeline, feature columns, and training metrics have been saved.")

if __name__ == "__main__":
    train_ensemble_model()