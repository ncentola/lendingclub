from helper_functions import setdiff
from xgboost import DMatrix
from model import Model
from numpy import nan
import pandas as pd
import pickle

class Scorer(object):

    def __init__(self, model, data):

        # if passed in a model object, set to self.model, else assume path was passed in and go load it
        if isinstance(model, Model):
            self.model = model
        else:
            try:
                if '.pkl' not in model:
                    model = model + '.pkl'

                with open(model, 'rb') as pickle_file:
                    self.model = pickle.load(pickle_file)
            except:
                raise ValueError('Must pass in Model object or valid path to a saved model folder')

        self.data = data

    def score(self, pred_contribs = False):
        model = self.model.fit_model
        scoring_data = self.data.modeling_data

        missing_cols = setdiff(self.model.train_columns, list(scoring_data.columns))
        extra_cols = setdiff(list(scoring_data.columns), self.model.train_columns)

        # print('Missing cols: ' + ', '.join(missing_cols))
        # print('Extra cols: ' + ', '.join(extra_cols))

        for col in missing_cols:
            if '__' in col:
                scoring_data[col] = 0
            else:
                scoring_data[col] = nan

        try:
            scoring_data = scoring_data.drop(extra_cols, axis = 1)
            print('Dropping ' + ', '.join(extra_cols))
        except:
            pass

        scoring_data = scoring_data[self.model.train_columns]
        xgb_data = DMatrix(scoring_data, label = self.data.target)

        if pred_contribs:

            contribs = model.predict(xgb_data, pred_contribs = True)
            self.contribs = pd.DataFrame.from_records(contribs, columns = list(scoring_data.columns) + ['bias'])

        self.preds = model.predict(xgb_data)
