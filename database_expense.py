import os
import streamlit as st  # pip install streamlit
from deta import Deta  # pip install deta
# from dotenv import load_dotenv

# load_dotenv(".env")
# Load the environment variables
DETA_KEY = "d0mnyrxagn7_mYggXzEZ9CoiAeauRxeaMum1NYX2x6bj"

print(DETA_KEY)
# Initialize with a project key
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("monthly_reports")

pers_fin_db = deta.Base(DETA_KEY)

def insert_period(period, incomes, expenses, comment):
    """Returns the report on a successful creation, otherwise raises an error"""
    return db.put({"key": period, "incomes": incomes, "expenses": expenses, "comment": comment})


def fetch_all_periods():
    """Returns a dict of all periods"""
    res = db.fetch()
    return res.items


def get_period(period):
    """If not found, the function will return None"""
    return db.get(period)


insert_period("2023_January",{"Salary":1500,"Blog":50,"Other Income":10},{"Rent":600,"Utilities":200,"Groceries":10,"Car":100,"Other Expenses":50,"Savings":70},"This is comment")