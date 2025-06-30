# OptimalFertilizers
lgbm_params = {
    "boosting_type": "gbdt",
    "device": "gpu",
    "colsample_bytree": 0.4366677273946288,
    "learning_rate": 0.016164161953515117,
    "max_depth": 12,
    "min_child_samples": 67,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 243,
    "random_state": 42,
    "reg_alpha": 6.38288560443373,
    "reg_lambda": 9.392999314379155,
    "subsample": 0.7989164499431718,
    "verbose": -1
}

xgb_params = {
    # https://www.kaggle.com/code/patrykszcz/simple-xgb-model-5-fold
    "device": "gpu",
    "max_depth": 12,
    "colsample_bytree": 0.467,
    "subsample": 0.86,
    "n_estimators": 10000,
    "learning_rate": 0.03,
    "gamma": 0.26,
    "max_delta_step": 4,
    "reg_alpha": 2.7,
    "reg_lambda": 1.4,
    "early_stopping_rounds": 100,
    "objective": 'multi:softprob',
    "random_state": 13,
    "enable_categorical": True,
}
