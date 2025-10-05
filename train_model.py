import pandas as pd
import joblib
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
    Trains the ensemble model on the cleaned, multi-mission dataset.
    This includes scaling, SMOTE for imbalance, and a soft-voting classifier.
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
    clf2 = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss') # mlogloss for multi-class
    clf3 = CatBoostClassifier(random_state=42, verbose=0, loss_function='MultiClass') # MultiClass for multi-class

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

    # --- Evaluate the Model ---
    print("\nEvaluating model performance on the test set...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    # Updated target names for the 3 classes
    target_names = ['False Positive', 'Candidate', 'Confirmed']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- Save the Assets ---
    joblib.dump(pipeline, 'exoplanet_ensemble_model.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    print("\nEnsemble pipeline and feature columns have been saved.")

if __name__ == "__main__":
    train_ensemble_model()