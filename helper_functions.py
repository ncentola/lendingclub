from sklearn.model_selection import cross_val_score
import numpy as np
import re

def regex_remove_cols(df, regex):
    r = re.compile(regex)
    filtered_cols = list(filter(r.match, list(df.columns)))
    return df.drop(filtered_cols, axis = 1)

def setdiff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def xgb_objective(params):
    print(params)
    base_score, max_depth, n_estimators, learning_rate, min_child_weight, gamma, colsample_bytree, reg_alpha, reg_lambda = params

    model.set_params(   base_score=base_score,
                        max_depth=max_depth,
                        n_estimators = n_estimators,
                        learning_rate=learning_rate,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        colsample_bytree = colsample_bytree,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda
                    )

    return -np.mean(cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring=tuning_metric, verbose= 11))

def set_objective_vars(model_in, X_in, y_in, tuning_metric_in):
    global model, tuning_metric
    global X, y

    model = model_in
    X = X_in
    y = y_in
    tuning_metric = tuning_metric_in
