from sklearn.model_selection import train_test_split
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from time import sleep
from copy import copy
import seaborn as sns
import pandas as pd
import numpy as np
import requests, os

class LendingClub():

    def __init__(self, config):

        self.config = config
        self.header = {'Authorization': self.config.auth_key, 'Content-Type': 'application/json'}

    def get_historic_data(self, webdriver_path='./chromedriver', destination_dir = 'historical_data'):
        if not os.file.exists(webdriver_path):
            raise('must specify valid path to webdriver for selenium')

        print('Getting historic data from lendingclub.com ...')

        drv = webdriver.Chrome(webdriver_path)
        drv.get('https://www.lendingclub.com/account/login.action')
        sleep(5)

        drv.find_element_by_name('email').send_keys(self.config.email)
        sleep(3)
        drv.find_element_by_name('password').send_keys(self.config.password)
        sleep(3)
        drv.find_element_by_name('password').send_keys(Keys.ENTER)
        sleep(5)

        drv.get('https://www.lendingclub.com/info/download-data.action')
        sleep(3)

        soup = BeautifulSoup(drv.page_source, 'html.parser')

        loan_period_dropdown_vals = [v['value'] for v in soup.find_all(id = 'loanStatsDropdown')[0].find_all('option')]

        select = Select(drv.find_element_by_id('loanStatsDropdown'))

        for val in loan_period_dropdown_vals:
            select.select_by_value(val)
            drv.find_element_by_id('currentLoanStatsFileName').click()
            sleep(30)

        drv.close()

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        zip_dir = os.path.join(destination_dir, 'zips')
        if not os.path.exists(zip_dir):
            os.makedirs(zip_dir)

        os.system('mv ~/Downloads/LoanStats* historical_data/')

        for file in os.listdir(destination_dir):
            try:
                z = ZipFile(os.path.join(destination_dir, file))
                z.extractall(path = destination_dir)
                os.system('mv historical_data/' + file + ' ' + zip_dir)
            except:
                print(file + ' not a zip')


    def build_historic_data(self, data_split_type, historic_data_dir = None):
        # if historic_data_dir is None:
        #     self.get_historic_data(destination_dir=historic_data_dir)

        dfs = []
        csvs = []
        for f in os.listdir(historic_data_dir):
            if '.csv' in f:
                df = pd.read_csv(historic_data_dir + f, skiprows=1, low_memory=False)

                # remove coborrower loans
                df = df.loc[df.application_type == 'Individual']

                # remove loans that no longer meet criteria
                df = df.loc[~df.loan_status.str.contains('Does not meet')]

                df['interest_num'] = pd.to_numeric(df.int_rate.str.replace('%', '')) /100
                df['roi'] = (df.total_pymnt_inv - df.funded_amnt_inv) / df.funded_amnt_inv
                df.loc[np.isnan(df.roi), 'roi'] = 0

                dfs.append(df)
                csvs.append(f)


        # create a df that shows which columns are completely NA for each file
        col_check = None
        for fname, df in list(zip(csvs, dfs)):
            all_missing = pd.DataFrame(df.isnull().all())
            all_missing.columns = [fname.replace('.csv', '')]
            all_missing = all_missing.transpose().reset_index(drop = False)
            if col_check is None:
                col_check = all_missing
            else:
                col_check = col_check.append(all_missing, ignore_index=True)

        # get all columns that have been exclude from some files but not others
        col_changes = col_check[col_check.columns[col_check.apply(pd.Series.nunique) != 1]]

        # remove columns where 3 or more files are completely missing it
        remove_cols = col_changes.drop('index', axis = 1).columns[list(col_changes.drop('index', axis = 1).apply(sum) >= 3)]

        # remove files with more than 25 missing cols
        for v in col_changes.index[col_changes.drop('index', axis = 1).apply(sum, axis = 1) > 25]:
            del dfs[v]

        sns.heatmap(col_changes.drop('index', axis = 1).replace({True: 1, False: 0}).transpose())

        # combine all files into master
        master_df = None
        for df in dfs:
            df = df.drop(remove_cols, axis = 1)
            if master_df is None:
                master_df = df
            else:
                master_df = master_df.append(df, ignore_index=True)

        master_df['issue_d'] = pd.to_datetime(master_df.issue_d)
        master_df['mob'] = (pd.datetime.now().date() - master_df.issue_d).dt.total_seconds() / 3600 / 24 / 30
        master_df['id'] = pd.to_numeric(master_df['id'])
        master_df['charge_off'] = master_df.loan_status == 'Charged Off'

        self.master_df = master_df

        mature_data = copy(master_df.loc[(master_df.mob >= 36) | (master_df.loan_status == 'Charged Off')])
        mature_data['term_numeric'] = pd.to_numeric(mature_data.term.str.replace(' months', ''))
        mature_data['oldest_cr_line_months'] = (pd.to_datetime(mature_data.issue_d) - pd.to_datetime(mature_data.earliest_cr_line)).dt.total_seconds() / 3600 / 24 / 30
        self.train_data, self.validation_data = self.build_train_and_validation(mature_data, split_type=data_split_type)

    def build_train_and_validation(self, df, validation_set_size = 0.3, split_type='stratified'):
        '''Split data using either stratified sampling or based on issue date'''
        if split_type == 'stratified':
            train_data, validation_data = train_test_split(df, test_size=validation_set_size, stratify=df.charge_off, random_state=42)

        if split_type == 'date':
            date_column = list(df.sort_values('issue_d', ascending=False)['issue_d'])
            index = range(0,len(date_column)+1)
            cutoff_date = date_column[np.int((np.percentile(index, validation_set_size * 100)))]

            train_data = df.loc[df.issue_d < cutoff_date]
            validation_data = df.loc[df.issue_d >= cutoff_date]

        return (train_data, validation_data)

    def get_fundable_loans(self):

        r = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing', headers=self.header, params={'showAll': 'true'})
        r.raise_for_status()

        fundable_loans = pd.DataFrame(r.json()['loans'])
        fundable_loans['roi'] = 0
        fundable_loans['charge_off'] = False
        self.fundable_loans = fundable_loans
