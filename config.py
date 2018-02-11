from configparser import ConfigParser

class Config():
    def __init__(self, path_to_config_file):
        cfp = ConfigParser()
        cfp.read(path_to_config_file)

        self.config = cfp

        self.investor_id = cfp.get('lending_club_account_data', 'investor_id')
        self.auth_key = cfp.get('lending_club_account_data', 'auth_key')
        self.email = cfp.get('lending_club_account_data', 'email')
        self.password = cfp.get('lending_club_account_data', 'password')

        self.aws_access_key_id = cfp.get('aws_credentials', 'aws_access_key_id')
        self.aws_secret_access_key = cfp.get('aws_credentials', 'aws_secret_access_key')
