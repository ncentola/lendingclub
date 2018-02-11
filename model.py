import helper_functions as hf
from modeling_data import ModelingData
from skopt import gp_minimize
from copy import copy
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle, os

class Model(object):
    def __init__(self, modeling_data):
        # if not isinstance(modeling_data, ModelingData):
            # raise ValueError('Must pass in ModelingData object via modeling_data arg')
        self.data = modeling_data
        # if isinstance(modeling_data, ModelingData):
        #     self.data = modeling_data
        # else:
        #     try:
        #         if '.pkl' not in modeling_data:
        #             modeling_data = os.path.join(data, '.pkl')
        #         with open(modeling_data, 'rb') as pickle_file:
        #             self.data = pickle.load(pickle_file)
        #     except:
        #         raise ValueError('Must pass in ModelingData object or valid path to a saved model folder')


        if self.data.target.dtype == 'bool':
            objective = 'binary:logistic'
            self.tuning_metric = 'roc_auc'
        else:
            objective = 'reg:linear'
            self.tuning_metric = 'neg_mean_absolute_error'

        self.best_params = { 'base_score': 0.5,
                             'booster': 'gbtree',
                             'colsample_bylevel': 1,
                             'colsample_bytree': 0.56747386852932058,
                             'gamma': 1.0827517721218596,
                             'learning_rate': 0.026769910899408403,
                             'max_delta_step': 0,
                             'max_depth': 6,
                             'min_child_weight': 37,
                             'missing': None,
                             'n_estimators': 86,
                             'n_jobs': -1,
                             'nthread': -1,
                             'objective': objective,
                             'random_state': 0,
                             'reg_alpha': 2,
                             'reg_lambda': 1,
                             'scale_pos_weight': 1,
                             'seed': 0,
                             'silent': True,
                             'subsample': 1}

    def tune_hyperparams(self, n_calls = 100, verbose = False, n_cores = 'auto'):

        if self.data.model_type == 'xgboost':

            X, y = self.data.modeling_data, self.data.target
            if self.data.target.dtype == 'bool':
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()

            param_space =   {
                                'base_score': [0.5],
                                'max_depth': (5, 10),
                                'n_estimators': (50, 125),
                                'learning_rate': (0.01, .1),
                                'min_child_weight': (1, 50),
                                'gamma': (0, 5.0),
                                'colsample_bytree': (0.50, .999),
                                'reg_alpha': (0, 5),
                                'reg_lambda': (0, 10)
                            }

            hf.set_objective_vars(model_in=model, X_in=X, y_in=y, tuning_metric_in=self.tuning_metric)
            gp_result = gp_minimize(hf.xgb_objective, dimensions=list(param_space.values()), n_calls=n_calls, random_state=0, verbose = 11, n_jobs = -1)

            tuned_params = dict(list(zip(param_space.keys(), gp_result.x)))
            default_params = model.get_params()

            for k, v in list(zip(tuned_params.keys(), tuned_params.values())):
                default_params[k] = v

            default_params['seed'] = 0
            default_params['nthread'] = 1

            self.gp_result = gp_result
            self.best_params = default_params

    def fit(self):
        train_data = self.data.modeling_data.drop(['id'], axis = 1)
        data = xgb.DMatrix(train_data, label = self.data.target)

        self.train_columns = list(train_data.columns)

        self.fit_model = xgb.train(params=self.best_params, dtrain = data, num_boost_round=self.best_params['n_estimators'])

    def save(self, model_name, dir = 'saved_models'):
        if not os.path.exists(dir):
            os.mkdir(dir)

        m = copy(self)

        del m.data

        with open(os.path.join(dir, model_name + '.pkl'), 'wb') as output:
            pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)
