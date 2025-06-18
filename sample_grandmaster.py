# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
import gc
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import time
from sklearn.base import BaseEstimator, TransformerMixin

df_sub = pd.read_csv("./data/sample_submission.csv")
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_extra = pd.read_csv('./data/extra.csv')

# Save IDs for later use
test_ids = df_test['id'].copy()

# Drop ID columns which aren't needed for modeling
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

# Combine training and extra data
df_train = pd.concat([df_train, df_extra], ignore_index=True)

# Display information about the dataset
df_train.info()

# Identify columns with too many missing values (>95%)
missing_threshold = 0.95
high_missing_columns = df_train.columns[df_train.isnull().mean() > missing_threshold]
if len(high_missing_columns) > 0:
    print(f"Dropping columns with >{missing_threshold*100}% missing values: {list(high_missing_columns)}")
    df_train = df_train.drop(columns=high_missing_columns)
    df_test = df_test.drop(columns=high_missing_columns)

# Print missing value information
missing_train = df_train.isna().mean() * 100
missing_values = df_train.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    print("\nMissing values in training data:")
    for col, count in missing_values.items():
        print(f"{col}: {count} ({count/len(df_train)*100:.2f}%)")
    
    plt.figure(figsize=(10, 6))
    plt.bar(x=missing_values.index, height=missing_values.values, color='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Missing Values')
    plt.title('Missing Values per Feature')
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found in the dataset.")

# Identify categorical and numerical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Fertilizer Name')  # Remove target from categorical columns
numerical_columns = df_train.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical columns: {categorical_columns}")
print(f"Numerical columns: {numerical_columns}")

# Create label encoder for target
le = LabelEncoder()
y = le.fit_transform(df_train['Fertilizer Name'])
X = df_train.drop(['Fertilizer Name'], axis=1)

def ratio_pipeline():
    return

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_columns)
    ],
    remainder='passthrough'
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit the preprocessor on training data
print("Preprocessing training data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Process test data
print("Preprocessing test data...")
X_test_processed = preprocessor.transform(df_test)

# Create and train the model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    n_estimators=3200,
    learning_rate=0.045,
    max_depth=7,
    colsample_bytree=0.6,
    colsample_bylevel=0.8,
    subsample=0.8,
)

print("Learning...")
model.fit(X_train_processed, y_train)

# Evaluate on validation set
y_pred_probs = model.predict_proba(X_val_processed)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
actual = [[label] for label in y_val]

def mapk(actual, predicted, k=3):
    """
    Compute Mean Average Precision at K (MAP@K)
    
    This function calculates the mean average precision at k metric,
    which is commonly used in recommender systems and information retrieval.
    
    """
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

map3_score = mapk(actual, top_3_preds)
print(f"✅ MAP@3 Score: {map3_score:.5f}")

# Make predictions on test data
test_probs = model.predict_proba(X_test_processed)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")

# Visualize feature importance
if hasattr(model, 'feature_importances_'):
    # Get feature names from the preprocessor
    feature_names = []
    
    # Add numerical feature names (these should remain the same after preprocessing)
    for name in numerical_columns:
        feature_names.append(name)
    
    # Add categorical feature names (these will be ordinal encoded)
    for name in categorical_columns:
        feature_names.append(name)
    
    # Create a dataframe with feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_[:len(feature_names)]
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 important features:")
    print(feature_importance.head(10))

# Example of creating a complete pipeline (preprocessor + model)
# This is just for demonstration - not used in the current analysis
def create_complete_pipeline():
    """Create a complete pipeline that includes preprocessing and the XGBoost classifier.
    
    Returns:
        Pipeline: A scikit-learn pipeline object containing the preprocessor and classifier.
    """
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            n_estimators=3200,
            learning_rate=0.045,
            max_depth=7,
            colsample_bytree=0.6,
            colsample_bylevel=0.8,
            subsample=0.8,
        ))
    ])
    return full_pipeline

# Custom Scorer for MAP@K
class MAP3Scorer:
    """Custom scorer class for Mean Average Precision at K (MAP@K) metric.
    
    This class creates a scorer compatible with scikit-learn's cross-validation
    functions that evaluates predictions using the MAP@K metric.
    """
    def __init__(self, k=3):
        self.k = k
        
    def __call__(self, estimator, X, y_true):
        # Predict probabilities
        y_proba = estimator.predict_proba(X)
        
        # Get top K predictions
        top_k_preds = np.argsort(y_proba, axis=1)[:, -self.k:][:, ::-1]
        
        # Format actual values for mapk function
        actual = [[label] for label in y_true]
        
        # Calculate MAP@K
        return mapk(actual, top_k_preds, k=self.k)

# Create a Make Scorer compatible function for MAP@K
def map_k_scorer(k=3):
    """Create a scikit-learn compatible scoring function for MAP@K.
    
    Parameters:
    -----------
    k : int, default=3
        Number of top predictions to consider
        
    Returns:
    --------
    callable
        A scoring function that can be used with scikit-learn's cross_val_score
    """
    map_scorer = MAP3Scorer(k=k)
    return make_scorer(map_scorer, greater_is_better=True)

# Cross-validation with MAP@K
def cv_with_mapk(X, y, model, preprocessor=None, n_splits=5, random_state=42):
    """Perform cross-validation with the MAP@K metric.
    
    Parameters:
    -----------
    X : DataFrame or array
        Features
    y : Series or array
        Target labels
    model : estimator
        The model to evaluate
    preprocessor : transformer, optional
        Preprocessing pipeline to apply before training
    n_splits : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Cross-validation results with scores and timing information
    """
    # Initialize the cross-validation strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize lists to store results
    fold_scores = []
    fold_times = []
    fold_models = []
    
    print(f"Starting {n_splits}-fold cross-validation with MAP@3 scoring...")
    
    # Iterate over folds
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        start_time = time.time()
        
        # Split data for this fold
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Preprocess if needed
        if preprocessor is not None:
            X_fold_train = preprocessor.fit_transform(X_fold_train)
            X_fold_val = preprocessor.transform(X_fold_val)
        
        # Fit the model
        print(f"Training fold {i+1}/{n_splits}...")
        model.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_pred_probs = model.predict_proba(X_fold_val)
        top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_fold_val]
        score = mapk(actual, top_3_preds)
        
        # Record results
        fold_time = time.time() - start_time
        fold_scores.append(score)
        fold_times.append(fold_time)
        fold_models.append(model)
        
        print(f"Fold {i+1} - MAP@3: {score:.5f} - Time: {fold_time:.2f}s")
    
    # Summarize results
    cv_results = {
        'scores': fold_scores,
        'mean_score': np.mean(fold_scores),
        'std_score': np.std(fold_scores),
        'times': fold_times,
        'mean_time': np.mean(fold_times),
        'models': fold_models
    }
    
    print(f"\nCross-Validation Summary:")
    print(f"Mean MAP@3: {cv_results['mean_score']:.5f} ± {cv_results['std_score']:.5f}")
    print(f"Mean training time: {cv_results['mean_time']:.2f}s per fold")
    
    return cv_results

# Add a function for model training with cross-validation
def train_with_cv(X, y, params=None, preprocessor=None, n_splits=5, random_state=42):
    """Train a model using cross-validation with MAP@K scoring.
    
    Parameters:
    -----------
    X : DataFrame or array
        Features
    y : Series or array
        Target labels
    params : dict, optional
        XGBoost model parameters
    preprocessor : transformer, optional
        Preprocessing pipeline to apply before training
    n_splits : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        The best model and cross-validation results
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'n_estimators': 3200,
            'learning_rate': 0.045,
            'max_depth': 7,
            'colsample_bytree': 0.6,
            'colsample_bylevel': 0.8,
            'subsample': 0.8,
        }
    
    # Create the model
    model = XGBClassifier(**params)
    
    # Run cross-validation
    cv_results = cv_with_mapk(
        X, y, 
        model=model, 
        preprocessor=preprocessor, 
        n_splits=n_splits, 
        random_state=random_state
    )
    
    # Train a final model on all data
    print("\nTraining final model on full dataset...")
    if preprocessor is not None:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = X
        
    final_model = XGBClassifier(**params)
    final_model.fit(X_processed, y)
    
    return final_model, cv_results

# Example usage of cross-validation pipeline
def run_cv_pipeline():
    """Example function showing how to use the cross-validation pipeline."""
    # Use the global variables X, y, and preprocessor defined earlier
    print("\nRunning cross-validation pipeline...")
    
    # Define model parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'n_estimators': 1000,  # Reduced for faster CV
        'learning_rate': 0.05,
        'max_depth': 7,
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.8,
        'subsample': 0.8,
        'random_state': 42
    }
    
    # Run cross-validation and train final model
    cv_model, cv_results = train_with_cv(
        X, y, 
        params=params, 
        preprocessor=preprocessor, 
        n_splits=5
    )
    
    # Process test data with final model
    print("Preprocessing test data...")
    X_test_processed = preprocessor.transform(df_test)
    
    # Make predictions on test data
    test_probs = cv_model.predict_proba(X_test_processed)
    top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
    top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,
        'Fertilizer Name': [' '.join(row) for row in top_3_labels]
    })
    submission.to_csv('submission_cv.csv', index=False)
    print("✅ Submission file saved as 'submission_cv.csv'")
    
    return cv_model, cv_results

# Function to run hyperparameter search with cross-validation
def hyperparameter_search(param_grid, n_splits=3):
    """
    Simple hyperparameter search using cross-validation.
    
    Parameters:
    -----------
    param_grid : dict of lists
        Grid of parameter values to search
    n_splits : int, default=3
        Number of CV folds for each parameter set
        
    Returns:
    --------
    dict
        Best parameters and their score
    """
    print("Starting hyperparameter search...")
    
    # Import product for parameter grid
    from itertools import product
    
    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combos = list(product(*param_values))
    
    # Track best parameters and score
    best_score = 0
    best_params = None
    
    # Create base parameters
    base_params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'random_state': 42,
    }
    
    # Loop through parameter combinations
    for i, combo in enumerate(param_combos):
        # Create parameter dictionary for this combo
        current_params = base_params.copy()
        for j, key in enumerate(param_keys):
            current_params[key] = combo[j]
        
        print(f"\nEvaluating parameters {i+1}/{len(param_combos)}: {current_params}")
        
        # Run CV with these parameters
        model = XGBClassifier(**current_params)
        cv_results = cv_with_mapk(X, y, model, preprocessor, n_splits=n_splits)
        
        # Check if this is the best score so far
        current_score = cv_results['mean_score']
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            print(f"✅ New best score: {best_score:.5f}")
    
    print("\nHyperparameter search complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best MAP@3 score: {best_score:.5f}")
    
    return {'best_params': best_params, 'best_score': best_score}

# Run the cross-validation pipeline if this script is executed directly
if __name__ == "__main__":
    # Set this to True to run the CV pipeline instead of the regular training
    RUN_CV = False
    
    # Set this to True to run hyperparameter search
    RUN_HYPERPARAM = False
    
    if RUN_CV:
        cv_model, cv_results = run_cv_pipeline()
    
    if RUN_HYPERPARAM:
        # Define parameter grid to search
        param_grid = {
            'n_estimators': [1000, 2000, 3000],
            'learning_rate': [0.03, 0.045, 0.06],
            'max_depth': [5, 7, 9],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        best = hyperparameter_search(param_grid, n_splits=3)
        
        # Train with best parameters
        print("\nTraining final model with best parameters...")
        final_model, _ = train_with_cv(X, y, params=best['best_params'], preprocessor=preprocessor)
