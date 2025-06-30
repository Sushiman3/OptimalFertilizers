from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

X = pd.read_csv('./data/original.csv', index_col='id')
y = X.pop('Fertilizer Name')
le = LabelEncoder()
y = le.fit_transform(y)
X = X.astype(str)

def adjusted_mutual_info(x, y, n_iter=5):
    x, y = x.astype(str), y.astype(str)
    m0 = mutual_info_score(x, y)
    m1 = Parallel(n_jobs=-1)(
        delayed(lambda rs: mutual_info_score(
            y, np.random.default_rng(rs).permutation(x)
        ))(rs)
        for rs in range(n_iter)
    )
    return m0 - np.mean(m1)

mi = {}
e =  mutual_info_score(y, y)
for c1, c2, c3 in tqdm(list(combinations(list(X.columns), 3))):
    c = c1+'_'+c2+'_'+c3
    mi[c] = adjusted_mutual_info(X[c1]+'_'+X[c2]+'_'+X[c3], y)/e

comb3 = sorted(mi, key=mi.get, reverse=True)

from sklearn.base import BaseEstimator, ClassifierMixin, clone

def Augmented(model, X_o, y_o, weight_arg='sample_weight', weight=1.0):
    class AugmentedModel(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            sample_weight = np.array([1.0]*len(X)+[weight]*len(X_o))
            X = pd.concat([X, X_o])
            y = np.concatenate([y, y_o])
            self.m = clone(model).fit(X, y, **{weight_arg: sample_weight})
            self.classes_ = self.m.classes_
            return self
        def predict_proba(self, X):
            return self.m.predict_proba(X)
    return AugmentedModel()

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression

X_o = pd.read_csv('./data/Fertilizer_Prediction.csv')
y_o = le.transform(X_o.pop('Fertilizer Name'))

X_all = pd.concat([X, X_o]).astype(str)

X_all_e = X_all.copy()
for c1, c2 in combinations(X_all.columns, 2):
    X_all_e[c1+'_'+c2] = X_all[c1]+'_'+X_all[c2]

topk = 10
for c1_c2_c3 in comb3[:topk]:
    c1, c2, c3 = c1_c2_c3.split('_')
    X_all_e[c1_c2_c3] = X_all[c1]+'_'+X_all[c2]+'_'+X_all[c3]

X_e = X_all_e.iloc[:len(X)]
X_o_e = X_all_e.iloc[len(X):]

def MAP(k):
    def mapk(y_true, y_pred):        
        y_pred = np.argsort(-y_pred, axis=1)[:, :k]
        m = (y_true[:, None] == y_pred)
        return np.mean(np.where(m.any(axis=1), 1/(m.argmax(axis=1)+1), 0))
    mapk.__name__ = F'MAP{k}'
    return mapk

MAP3 = MAP(3)
scorer = make_scorer(MAP3, response_method='predict_proba')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

model = Augmented(
    make_pipeline(
        OneHotEncoder(handle_unknown='ignore'),
        LogisticRegression(C=1e-2, max_iter=1000, random_state=0)
    ), X_o_e, y_o, weight_arg='logisticregression__sample_weight', 
    weight=4.0
)

scores = cross_val_score(
    model, X_e, y, scoring=scorer,
    cv=kfold, n_jobs=4,
    error_score='raise'
)
print(F'{scores.mean():.5f} ± {scores.std():.5f}')
