import pandas as pd

INPUT_FILE = 'df_final_cleaned.csv'
OUTPUT_FILE = 'exoplanet_data_merged_for_ensemble.csv'

def prepare_merged_data_for_ensemble():
    """
    Loads the user's custom merged dataset, selects the correct features,
    cleans it, and prepares it for the ensemble model.
    """
    print(f"Loading custom merged data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"--- FATAL ERROR ---")
        print(f"File not found: '{INPUT_FILE}'.")
        print("Please make sure your merged dataset is in the correct folder and named correctly.")
        return

    # --- Feature Selection ---
    # These are the columns we will use as features for the model
    feature_columns = [
        'period_days', 'duration_hours', 'depth_ppm', 'planet_radius_rearth',
        'stellar_teff', 'stellar_radius_solar', 'stellar_logg'
    ]
    
    # This is our target variable
    target_column = 'disposition'

    # Columns to keep (features + target)
    columns_to_keep = feature_columns + [target_column]

    print(f"Original dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Check if all required columns exist
    if not all(col in df.columns for col in columns_to_keep):
        print("--- FATAL ERROR ---")
        print("Your CSV is missing one or more required columns.")
        print(f"Required columns are: {columns_to_keep}")
        return

    df_clean = df[columns_to_keep].copy()
    print(f"Selected {len(feature_columns)} features for the model.")

    # --- Data Cleaning and Preprocessing ---
    # 1. Handle missing values
    initial_rows = len(df_clean)
    df_clean.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(df_clean)} rows with missing values.")

    # 2. Encode the 3-class target variable
    # We will map text labels to numbers: 0, 1, 2
    disposition_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
    
    # Ensure all dispositions are valid before mapping
    df_clean = df_clean[df_clean['disposition'].isin(disposition_map.keys())]
    df_clean['disposition'] = df_clean['disposition'].map(disposition_map)
    
    # Save the final, clean dataset
    df_clean.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nData preparation complete. Cleaned merged data saved to '{OUTPUT_FILE}'.")
    print(f"Final dataset has {df_clean.shape[0]} rows.")
    print(f"Final disposition counts:\n{df_clean['disposition'].value_counts()}")

if __name__ == "__main__":
    prepare_merged_data_for_ensemble()