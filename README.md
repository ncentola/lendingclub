# Lending Club Loan Analysis & Modeling

A set of tools to build an implement predictive models on Lending Club historic and API data

## Getting Started

Some things that need to be done before we start trying to pull data and build models

### Prerequisites

* A valid Lending Club investor account
* Some sort of config file that looks like
```
[lending_club_account_data]
email: YOUR_LENDING_CLUB_EMAIL
password: YOUR_LENDING_CLUB_PASSWORD
investor_id: YOUR_INVESTOR_ID
auth_key: YOUR_API_KEY
```
* If you plan on using the get_historic_data method in the LendingClub class you must have a valid web driver for selenium to use. The default assumption is that there is a chromedriver in the root directory
