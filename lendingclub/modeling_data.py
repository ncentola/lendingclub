from lendingclub.helper_functions import regex_remove_cols
from sklearn.model_selection import train_test_split
from lendingclub.lendingclub import LendingClub
from sklearn.ensemble import IsolationForest
from copy import copy
import pandas as pd
import numpy as np

class ModelingData():

    def __init__(self, data, target, model_type = 'xgboost'):

        self.target_column = target
        self.target = data[target]
        self.model_type = model_type
        self.raw = data


    def build_train_and_validation(self, lending_club, validation_set_size, cols_to_drop=['roi', 'loan_status']):
        mature_data = copy(lending_club.mature_data)
        target = mature_data.target

        # try to drop everything in cols_to_drop and target variable
        for col in cols_to_drop + [target]:
            try:
                mature_data = mature_data.drop(col, axis=1)
            except:
                pass

        train_data, validation_data, train_label, validation_label = train_test_split(mature_data, target, test_size=validation_set_size, stratify=target, random_state=42)

    def build(self, ordinal_risk = False, targets_to_drop=['roi', 'loan_status', 'charge_off']):
        modeling_data = copy(self.raw)
        modeling_data['id'] = pd.to_numeric(modeling_data.id)
        modeling_data['revol_util'] = pd.to_numeric(modeling_data.revol_util.str.replace('%', '')) / 100
        modeling_data['revol_mult'] = modeling_data.revol_util * modeling_data.revol_bal
        modeling_data['pti'] = modeling_data.installment / modeling_data.annual_inc / 12
        modeling_data['zip_code'] = pd.to_numeric(modeling_data.zip_code.str.replace('xx', ''))
        modeling_data['employment_years'] = pd.to_numeric(modeling_data.emp_length.str.replace(r'[^0-9]', ''))

        if ordinal_risk:
            sorted_risk = sorted(list(modeling_data.sub_grade.unique()))
            risk_ordinal = pd.DataFrame(list(zip(sorted_risk, np.arange(0, len(sorted_risk)))), columns = ['sub_grade', 'risk_grade_rank'])

            modeling_data = pd.merge(modeling_data, risk_ordinal, on = ['sub_grade'])

            cols_to_drop = ['grade', 'sub_grade', 'addr_state', 'annual_inc_joint', 'application_type', 'collection_recovery_fee', 'debt_settlement_flag', 'debt_settlement_flag_date', 'dti_joint', 'earliest_cr_line', 'emp_length', 'emp_title', 'funded_amnt_inv', 'int_rate', 'issue_d', 'loan_amnt', 'member_id', 'mob', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'policy_code', 'pymnt_plan', 'recoveries', 'term', 'title', 'payment_plan_start_date', 'orig_projected_additional_accrued_interest', 'url', 'verification_status_joint', 'revol_bal_joint', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'total_pymnt_inv', 'total_pymnt', 'term_numeric']
        else:
            cols_to_drop = ['addr_state', 'annual_inc_joint', 'application_type', 'collection_recovery_fee', 'debt_settlement_flag', 'debt_settlement_flag_date', 'dti_joint', 'earliest_cr_line', 'emp_length', 'emp_title', 'funded_amnt_inv', 'int_rate', 'issue_d', 'loan_amnt', 'member_id', 'mob', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'policy_code', 'pymnt_plan', 'recoveries', 'term', 'title', 'payment_plan_start_date', 'orig_projected_additional_accrued_interest', 'url', 'verification_status_joint', 'revol_bal_joint', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'total_pymnt_inv', 'total_pymnt', 'term_numeric']


        regex_remove = ['_joint', 'last_', 'sec_', 'hardship_', 'settlement']

        for r in regex_remove:
            modeling_data = regex_remove_cols(modeling_data, r)

        for col in cols_to_drop + targets_to_drop + [self.target_column]:
            try:
                modeling_data = modeling_data.drop(col, axis = 1)
            except:
                pass

        self.modeling_data = pd.get_dummies(modeling_data, prefix_sep = '__')

    def remove_outliers(self, ordinal_risk = False, contamination = .1):
        if ordinal_risk:
            clustering_features = ['id', 'funded_amnt', 'installment', 'annual_inc', 'revol_mult', 'risk_grade_rank']
        else:
            clustering_features = ['id', 'funded_amnt', 'installment', 'annual_inc', 'revol_mult', 'grade__A', 'grade__B', 'grade__C', 'grade__D', 'grade__E', 'grade__F', 'grade__G']

        clustering_data = copy(self.modeling_data[clustering_features])
        clustering_data['roi'] = copy(self.target)
        clustering_data = clustering_data.dropna()

        clf = IsolationForest(max_samples=100, random_state=42, contamination=contamination)
        clf.fit(clustering_data)

        pred = clf.predict(clustering_data)
        non_outliers = clustering_data.loc[pred == 1]

        outliers = clustering_data.loc[pred == -1]

        print('Removing ' + str(len(outliers.id)) + ' outlier rows.')

        self.target = self.target.loc[list(self.modeling_data.id.isin(non_outliers.id))]
        self.modeling_data = self.modeling_data.loc[list(self.modeling_data.id.isin(non_outliers.id))]

        self.outliers = outliers

    def save(self, dir = 'saved_data'):
            if not os.path.exists(dir):
                os.mkdir(dir)

            if '.pkl' not in self.data_name:
                self.data_name = self.data_name + '.pkl'

            with open(os.path.join(dir, self.data_name), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
