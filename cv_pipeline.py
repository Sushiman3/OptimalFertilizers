"""
Cross-validation Pipeline for Optimal Fertilizers Competition

This script demonstrates how to use the cross-validation pipeline for the
XGBoost model with MAP@K scoring.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
print("Loading data...")
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

# Define MAP@K function
def mapk(actual, predicted, k=3):
    """
    Compute Mean Average Precision at K (MAP@K)
    
    This function calculates the mean average precision at k metric,
    which is commonly used in recommender systems and information retrieval.
    
    Parameters:
    -----------
    actual : list of lists
        Ground truth labels, each inner list contains the relevant items for a query
    predicted : list of lists
        Predicted labels, each inner list contains the predicted items for a query
    k : int, default=3
        The maximum number of predicted elements
        
    Returns:
    --------
    float
        The mean average precision at k
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

# Custom Scorer for MAP@K
class MAP3Scorer:
    """Custom scorer class for Mean Average Precision at K (MAP@K) metric."""
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

# Handle missing data and identify column types
print("Preprocessing data...")
missing_threshold = 0.95
high_missing_columns = df_train.columns[df_train.isnull().mean() > missing_threshold]
if len(high_missing_columns) > 0:
    print(f"Dropping columns with >{missing_threshold*100}% missing values: {list(high_missing_columns)}")
    df_train = df_train.drop(columns=high_missing_columns)
    df_test = df_test.drop(columns=high_missing_columns)

# Identify categorical and numerical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Fertilizer Name')  # Remove target from categorical columns
numerical_columns = df_train.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical columns: {len(categorical_columns)}")
print(f"Numerical columns: {len(numerical_columns)}")

# Create label encoder for target
le = LabelEncoder()
y = le.fit_transform(df_train['Fertilizer Name'])
X = df_train.drop(['Fertilizer Name'], axis=1)

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

def cv_with_mapk(X, y, model, preprocessor=None, n_splits=5, random_state=42):
    """Perform cross-validation with the MAP@K metric."""
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
            print(f"  Preprocessing fold {i+1}...")
            X_fold_train = preprocessor.fit_transform(X_fold_train)
            X_fold_val = preprocessor.transform(X_fold_val)
        
        # Fit the model
        print(f"  Training fold {i+1}/{n_splits}...")
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
        
        print(f"  Fold {i+1} - MAP@3: {score:.5f} - Time: {fold_time:.2f}s")
    
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

def train_with_cv(X, y, params=None, preprocessor=None, n_splits=5, random_state=42):
    """Train a model using cross-validation with MAP@K scoring."""
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'n_estimators': 1000,  # Reduced for CV
            'learning_rate': 0.05,
            'max_depth': 7,
            'colsample_bytree': 0.6,
            'colsample_bylevel': 0.8,
            'subsample': 0.8,
            'random_state': random_state,
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

def run_cv_pipeline():
    """Run the cross-validation pipeline and generate submission."""
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

if __name__ == "__main__":
    # Choose which operation to run
    RUN_CV = False          # Run cross-validation
    RUN_HYPERPARAM = True  # Run hyperparameter search
    
    if RUN_CV:
        cv_model, cv_results = run_cv_pipeline()
    
    if RUN_HYPERPARAM:
        # Define parameter grid to search
        param_grid = {
            'n_estimators': [500, 1000],
            'learning_rate': [0.03, 0.05],
            'max_depth': [5, 7],
            'subsample': [0.7, 0.8]
        }
        
        best = hyperparameter_search(param_grid, n_splits=3)
        
        # Train with best parameters
        print("\nTraining final model with best parameters...")
        final_model, _ = train_with_cv(X, y, params=best['best_params'], preprocessor=preprocessor)
        
        # Process test data
        X_test_processed = preprocessor.transform(df_test)
        
        # Make predictions
        test_probs = final_model.predict_proba(X_test_processed)
        top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
        top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': [' '.join(row) for row in top_3_labels]
        })
        submission.to_csv('submission_best.csv', index=False)
        print("✅ Best model submission saved as 'submission_best.csv'")
