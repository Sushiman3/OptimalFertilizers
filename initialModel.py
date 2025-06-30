# Map@3 Scpre: 0.34331

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

full_train_data = pd.read_csv('./data/train.csv') # 訓練データ
full_test_data = pd.read_csv('./data/test.csv')   # テストデータ
extra = pd.read_csv('./data/extra.csv')  # 追加データ
# Concatenate train and extra data
full_train_data = pd.concat([full_train_data, extra], ignore_index=True)

def column_ratio(X):
    return X[:, [0]] / (X[:,[0]] + X[:, [1]] + X[:,[2]])


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def app_temp(X):
    t = X[:, [0]]
    h = X[:, [1]]
    return t - (t - 10) * (0.8 - h / 100) / 2.3


def app_temp_name(function_transformer, feature_names_in):
    return ["temp"]


def app_temp_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(app_temp, feature_names_out=app_temp_name),
        StandardScaler(),
    )


categorical_columns = full_train_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Fertilizer Name')  # Remove target from categorical columns
numerical_columns = full_train_data.select_dtypes(include=['number']).columns.tolist()
numerical_columns.remove('id')  # Remove 'id' from numerical columns

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

le = LabelEncoder()
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
)

preprocessor = ColumnTransformer(
    [
        ("app_temp", app_temp_pipeline(), ["Temparature", "Humidity"]),
        ("num", num_pipeline, numerical_columns),
        ("cat", cat_pipeline, categorical_columns),
        ("nit_pot_ratio", ratio_pipeline(), ["Nitrogen", "Potassium","Phosphorous"]),
        ("nit_pho_ratio", ratio_pipeline(), ["Potassium","Nitrogen", "Phosphorous"]),
        ("pot_pho_ratio", ratio_pipeline(), ["Phosphorous","Potassium", "Nitrogen"]),
    ]
)

X = full_train_data.drop(columns=['id', 'Fertilizer Name'])
y = le.fit_transform(full_train_data['Fertilizer Name'])

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

"""
X_prepared_df = pd.DataFrame(
    X_prepared, columns=preprocessing.get_feature_names_out(), index=X.index
)
"""

def mapk(actual, predicted, k=3):
    #Compute Mean Average Precision at K (MAP@K)
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

model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    n_estimators=3200,
    learning_rate=0.045,
    max_depth=7,
    colsample_bytree=0.6,
    colsample_bylevel=0.8,
    subsample=0.8,
    tree_method='hist',
    device = 'cuda',
)

print("Learning...")
model.fit(X_train_processed, y_train)

# Evaluate on validation set
y_pred_probs = model.predict_proba(X_val_processed)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
actual = [[label] for label in y_val]

map3_score = mapk(actual, top_3_preds)
print(f"✅ MAP@3 Score: {map3_score:.5f}")

# Make predictions on test data
X_test_processed = preprocessor.transform(full_test_data.drop(columns=['id']))
test_ids = full_test_data['id']

test_probs = model.predict_proba(X_test_processed)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'initialsubmission.csv'")