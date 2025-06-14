# 基本ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 不要と思われる警告を消す
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 機械学習用ライブラリ
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

full_train_data = pd.read_csv('./data/train.csv') # 訓練データ
full_test_data = pd.read_csv('./data/test.csv')   # テストデータ

full_train_data # 訓練データの表示

full_test_data # テストデータのデータの表示

X = full_train_data.drop(columns=['id', 'Fertilizer Name'])
y = full_train_data['Fertilizer Name']

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state=42)

standard_scaler = StandardScaler()
ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()
num_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'] # 数値データの列
cat_columns = ['Soil Type', 'Crop Type'] # 文字列データの列
train_X[num_columns] = standard_scaler.fit_transform(train_X[num_columns])
val_X[num_columns] = standard_scaler.transform(val_X[num_columns])
train_X[cat_columns] = ordinal_encoder.fit_transform(train_X[cat_columns])
val_X[cat_columns] = ordinal_encoder.transform(val_X[cat_columns])
train_y = label_encoder.fit_transform(train_y)
val_y = label_encoder.transform(val_y)

model = XGBClassifier(
    learning_rate = 0.3,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 1.0,
    colsample_bytree = 1.0,
    n_estimators = 10000, # どこまで学習を進めるか。途中終了するため、どれだけ大きくても良い。
    objective='multi:softprob',
    num_class=7, # 答えとなり得る肥料は7種類
    eval_metric='mlogloss',
    use_label_encoder=False,
    early_stopping_rounds=100, #100回連続で精度が上がらなければ終了
    random_state=42
)

model.fit(
    train_X, train_y,
    eval_set=[(train_X, train_y), (val_X, val_y)],
    verbose=True
)

plt.ylim((1.85, 1.95))
sns.lineplot(model.evals_result()['validation_0']['mlogloss'], label='train')
sns.lineplot(model.evals_result()['validation_1']['mlogloss'], label='val')

def mapk(true_y, pred_y, k=3):
    score = 0.0
    for i in range(true_y.shape[0]):
        true = true_y[i]
        pred_proba = pred_y[i]
        pred = np.argsort(pred_proba)
        for rank in range(1, 4):
            if true == pred[-rank]:
                score += 1.0 / rank
                break
    return score / true_y.shape[0]

pred_val = model.predict_proba(val_X)
mapk(val_y, pred_val)

plt.figure(figsize=(10,4))
sns.barplot(x=X.columns, y=model.feature_importances_)

# テストデータの形を訓練データに一致させる
test_X = full_test_data.drop(columns="id")
test_X[num_columns] = standard_scaler.transform(test_X[num_columns])
test_X[cat_columns] = ordinal_encoder.transform(test_X[cat_columns])

# 予測を行う
pred = model.predict_proba(test_X)

pd.DataFrame(pred, index=full_test_data.id, columns=label_encoder.classes_)

first = label_encoder.inverse_transform(np.argsort(pred)[:, -1])
second = label_encoder.inverse_transform(np.argsort(pred)[:, -2])
third = label_encoder.inverse_transform(np.argsort(pred)[:, -3])

submission = pd.read_csv('./data/sample_submission.csv')  # 提出ファイルのテンプレート
submission # 提出ファイルのテンプレート

submission['Fertilizer Name'] = first + ' ' + second + ' ' + third
submission.to_csv('submission.csv', index=False)
submission

print(model.feature_importances_)


