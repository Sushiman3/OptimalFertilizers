# Optimal Fertilizers Competition - Cross-Validation Pipeline

This repository contains a solution for the Optimal Fertilizers competition, including a custom cross-validation pipeline with MAP@K scoring.

## Files

- `sample_grandmaster.py` - Original model implementation with added cross-validation functionality
- `cv_pipeline.py` - Standalone cross-validation pipeline implementation
- `data/` - Directory containing competition data
  - `train.csv` - Training data
  - `test.csv` - Test data
  - `extra.csv` - Additional training data
  - `sample_submission.csv` - Example of submission format

## Cross-Validation Pipeline Features

The cross-validation pipeline implemented in this project includes:

1. **Custom MAP@K Scoring**: Implementation of Mean Average Precision at K metric (MAP@K), which is the competition's evaluation metric.
2. **Stratified K-Fold**: Ensures balanced class distribution across folds.
3. **Preprocessing Pipeline**: Handles missing values, numerical scaling, and categorical encoding consistently across folds.
4. **Hyperparameter Search**: Optional grid search capability to find optimal model parameters.
5. **Timing Information**: Tracks execution time for each fold and overall training.

## How to Use

### Basic Cross-Validation

Run the standalone CV pipeline:

```bash
python cv_pipeline.py
```

This will:
1. Load and preprocess the data
2. Run 5-fold cross-validation with MAP@3 scoring
3. Train a final model on the full dataset
4. Generate predictions for the test set
5. Create a submission file (`submission_cv.csv`)

### Hyperparameter Tuning

To run hyperparameter tuning:

1. Open `cv_pipeline.py`
2. Set `RUN_HYPERPARAM = True` 
3. Adjust the parameter grid in the `param_grid` dictionary
4. Run the script:

```bash
python cv_pipeline.py
```

The best parameters will be used to train a final model and generate a submission file (`submission_best.csv`).

### Using the Cross-Validation Functions in Your Code

You can import the CV functions into your own code:

```python
from cv_pipeline import cv_with_mapk, train_with_cv

# Define your model parameters
params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y)),
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'random_state': 42
}

# Run cross-validation and get a trained model
model, cv_results = train_with_cv(
    X, y, 
    params=params, 
    preprocessor=my_preprocessor, 
    n_splits=5
)

# Check the cross-validation results
print(f"Mean MAP@3 score: {cv_results['mean_score']:.5f}")
```

## The MAP@K Metric

The Mean Average Precision at K (MAP@K) metric is specifically designed for ranking tasks where we need to predict the top K items. It rewards models for:

1. Correctly identifying relevant items
2. Ranking relevant items higher in the prediction list

For this competition, K=3, meaning we predict the top 3 fertilizer names for each soil sample.

### Formula

The MAP@K is calculated as:

1. For each query (soil sample), calculate the Average Precision (AP):
   - Consider only the first K predictions
   - For each correct prediction at position i, add 1/i to the score
   - Normalize by the minimum of K and the number of relevant items

2. Take the mean of the AP scores across all queries

## Implementation Details

The cross-validation pipeline handles:

- Preprocessing data consistently across folds
- Tracking performance metrics for each fold
- Training a final model on the complete dataset
- Creating appropriate submission files