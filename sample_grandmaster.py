# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import gc
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

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

# Example of how you could use the complete pipeline for model tuning
# from sklearn.model_selection import GridSearchCV
# pipeline = create_complete_pipeline()
# param_grid = {
#     'classifier__max_depth': [5, 7, 9],
#     'classifier__learning_rate': [0.01, 0.05, 0.1]
# }
# grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")
# best_model = grid_search.best_estimator_