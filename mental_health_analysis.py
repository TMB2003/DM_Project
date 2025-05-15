import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. Load Data ---
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    return df

# --- 2. Preprocess Data ---
def preprocess_data(df, target_column='mental_health_risk'):
    print("\n--- Starting Data Preprocessing ---")
    
    # Handle missing values by dropping rows with any NaN
    print(f"\nOriginal dataset shape: {df.shape}")
    df.dropna(inplace=True)
    print(f"Shape after dropping NaNs: {df.shape}")
    
    if df.empty:
        raise ValueError("Dataset is empty after dropping NaNs. Please check your data.")

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y_raw = df[target_column]
    
    # Encode target variable
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y_raw)
    print(f"\nTarget variable '{target_column}' encoded.")
    print(f"Target classes: {label_encoder_y.classes_} -> {np.unique(y)}")
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['number']).columns
    
    print(f"\nCategorical features: {list(categorical_features)}")
    print(f"Numerical features: {list(numerical_features)}")
    
    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Should not be needed due to dropna, but good practice
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Should not be needed
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create a column transformer to apply pipelines to respective columns
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough') # passthrough for any columns not specified
    
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding for better interpretability if needed
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError: # Older sklearn versions
        # Manual reconstruction for older scikit-learn versions
        feature_names_out = list(numerical_features) 
        ohe_categories = preprocessor.named_transformers_['cat']['onehot'].categories_
        for i, col in enumerate(categorical_features):
            for cat_val in ohe_categories[i]:
                feature_names_out.append(f"{col}_{cat_val}")

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names_out, index=X.index)
    print("\nData preprocessing complete. Features processed and scaled.")
    print("Processed features shape:", X_processed_df.shape)

    return X_processed_df, y, label_encoder_y, X # Return original X for sample testing

# --- 3. Classification Models ---
def train_and_evaluate_classification(X_processed, y, label_encoder_y):
    print("\n--- Classification Models ---")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing data shape: X_test {X_test.shape}, y_test {y_test.shape}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {accuracy:.4f}")
        trained_models[name] = model
        
    return trained_models

# --- 4. Clustering Models ---
def apply_clustering_models(X_processed):
    print("\n--- Clustering Models ---")
    # Note: For clustering, we typically use the full dataset (X_processed)
    # and do not use target labels (y).
    
    # KMeans Clustering
    print("\nApplying KMeans Clustering...")
    # Determine number of clusters, e.g., based on unique values in original target or elbow method
    # For this example, we assume 3 clusters (Low, Medium, High risk)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X_processed)
    print(f"KMeans cluster assignments (first 10 samples): {kmeans_labels[:10]}")
    print(f"Number of data points in each KMeans cluster: {np.bincount(kmeans_labels)}")

    # DBSCAN Clustering
    # DBSCAN is sensitive to scale and parameters (eps, min_samples).
    # We already scaled numerical features. Default parameters might not be optimal.
    print("\nApplying DBSCAN Clustering...")
    # Using default eps=0.5, min_samples=5. May need tuning.
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_processed)
    print(f"DBSCAN cluster assignments (first 10 samples): {dbscan_labels[:10]}")
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_ = list(dbscan_labels).count(-1)
    print(f"Estimated number of DBSCAN clusters: {n_clusters_}")
    print(f"Estimated number of noise points in DBSCAN: {n_noise_}")
    
    return {"KMeans": kmeans, "DBSCAN": dbscan}

# --- 5. Test Models with Samples ---
def test_models_with_samples(df_original, X_original_unencoded, y_original_encoded, 
                             label_encoder_y, trained_classification_models, 
                             trained_clustering_models, preprocessor, num_samples=5):
    print("\n--- Testing Models with Samples ---")
    
    if X_original_unencoded.empty:
        print("No samples to test as the dataset became empty after preprocessing.")
        return

    # Ensure num_samples is not greater than available data
    num_samples = min(num_samples, len(X_original_unencoded))
    if num_samples == 0:
        print("No samples to test.")
        return
        
    sample_indices = X_original_unencoded.sample(n=num_samples, random_state=42).index
    
    # Get original unencoded features for these samples
    X_samples_unencoded = X_original_unencoded.loc[sample_indices]
    
    # Preprocess these specific samples for model input
    X_samples_processed = preprocessor.transform(X_samples_unencoded)
    X_samples_processed_df = pd.DataFrame(X_samples_processed, columns=preprocessor.get_feature_names_out(), index=X_samples_unencoded.index)
    
    # Get true labels for these samples
    y_true_encoded_samples = y_original_encoded[sample_indices]
    y_true_decoded_samples = label_encoder_y.inverse_transform(y_true_encoded_samples)
    
    print(f"\nShowing results for {num_samples} random samples:")
    
    for i in range(num_samples):
        idx = sample_indices[i]
        print(f"\nSample #{i+1} (Original Index: {idx})")
        print(f"  Original Features: \n{X_samples_unencoded.iloc[i]}")
        print(f"  True Mental Health Risk: {y_true_decoded_samples[i]}")
        
        # Classification predictions
        print("  Classification Predictions:")
        for name, model in trained_classification_models.items():
            pred_encoded = model.predict(X_samples_processed_df.iloc[i:i+1])
            pred_decoded = label_encoder_y.inverse_transform(pred_encoded)
            print(f"    {name}: {pred_decoded[0]}")
            
        # Clustering predictions
        print("  Clustering Assignments:")
        kmeans_cluster = trained_clustering_models["KMeans"].predict(X_samples_processed_df.iloc[i:i+1])
        print(f"    KMeans Cluster: {kmeans_cluster[0]}")
        
        # For DBSCAN, predict might not be available or meaningful in the same way as KMeans for new points if not part of fit.
        # We'll show the label it received during the fit process if the sample was in the training set for clustering.
        # Or, for a general case, one might re-fit or use a more complex approach to assign new points to DBSCAN clusters.
        # Here, we are using predict which is available in scikit-learn's DBSCAN, but its behavior can vary.
        dbscan_cluster = trained_clustering_models["DBSCAN"].fit_predict(X_samples_processed_df.iloc[i:i+1]) # fit_predict on single sample is more like assigning
        print(f"    DBSCAN Cluster: {dbscan_cluster[0]} (Note: DBSCAN prediction on single new points can be tricky)")

# --- Main Execution --- 
if __name__ == '__main__':
    filepath = 'mental_health_dataset.csv' # Make sure this file is in the same directory or provide full path
    
    # 1. Load Data
    df = load_data(filepath)
    
    # 2. Preprocess Data
    # We need the original X before one-hot encoding for sample display, and original df for reference
    X_processed_df, y_encoded, label_encoder_y, X_original_unencoded_for_samples = preprocess_data(df.copy()) # Use df.copy() to avoid SettingWithCopyWarning later
    
    # 3. Train and Evaluate Classification Models
    trained_classifiers = train_and_evaluate_classification(X_processed_df, y_encoded, label_encoder_y)
    
    # 4. Apply Clustering Models
    # Clustering is done on all processed features (X_processed_df)
    trained_clusterers = apply_clustering_models(X_processed_df.copy()) # Use .copy() if DBSCAN modifies input
    
    # 5. Test Models with Samples
    # For testing, we use the original unencoded features (X_original_unencoded_for_samples) 
    # and the overall preprocessor to transform them just before prediction.
    # The y_encoded corresponds to the full dataset after cleaning.
    
    # Re-access the preprocessor object from the preprocess_data function scope
    # This is a bit of a workaround for a standalone script. In a notebook, cells would share scope more easily.
    # For a cleaner script, preprocessor could be returned and passed around.
    # Re-creating the preprocessor definition here for clarity for `test_models_with_samples`
    
    # Re-define preprocessor to pass to test_models_with_samples
    # (This implicitly assumes that the columns and dtypes are consistent, which they are because we use df.copy())
    temp_X = df.drop('mental_health_risk', axis=1)
    temp_X.dropna(inplace=True) # Ensure it's in the same state as X_original_unencoded_for_samples
    categorical_features_test = temp_X.select_dtypes(include=['object', 'category']).columns
    numerical_features_test = temp_X.select_dtypes(include=['number']).columns

    numerical_pipeline_test = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline_test = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor_for_testing = ColumnTransformer([
        ('num', numerical_pipeline_test, numerical_features_test),
        ('cat', categorical_pipeline_test, categorical_features_test)
    ], remainder='passthrough')
    preprocessor_for_testing.fit(X_original_unencoded_for_samples) # Fit on the same data used for X_processed_df

    test_models_with_samples(df, X_original_unencoded_for_samples, y_encoded, 
                             label_encoder_y, trained_classifiers, 
                             trained_clusterers, preprocessor_for_testing, num_samples=5)

    print("\n--- Script Execution Finished ---")
