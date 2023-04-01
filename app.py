import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import database as db

# --- USER AUTHENTICATION ---
users = db.fetch_all_users()

usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]


credentials= {}
credentials["usernames"] = {}
for username,name,password in zip(usernames,names,hashed_passwords):
    credentials["usernames"][username] = {"name":name,"password":password}

# print(credentials)

authenticator = stauth.Authenticate(credentials,
    "sales_dashboard", "abcdef", cookie_expiry_days=30,preauthorized=True)

try:
    if authenticator.register_user('Register user',preauthorization=True):
        st.title("👋 Personal Finance")

        name = st.text_input("Name:")
        no_of_deps = st.number_input("Number of dependents:",0,10,step=1)
        age = st.multiselect("What is your age?",("under 30 years","31 to 39 years","40 to 50 years","Over 50 years"))
        experience = st.multiselect("Investing Experience?",("I have never invested before","1 to 3 years","4 to 7 years","More than 7 years"))
        dependants = st.multiselect("Who depends on you finncially?",("Kid(s)","Spouse","Parents","Other family","No dependents"))
        risk_app = st.multiselect("How much of a temporary drop in portfolio value could you handle?",("I can't take any drop","A drop between 5-10%","A drop between 10-20%","Greater than 20%"))
        dependants = st.multiselect("How do you want to invest your money?",("Balanced: Focussed on higher returns while minimising risk",
                                                                            "Growth: Focussed on superior long-term returns with moderate risk",
                                                                            "Aggressive: High-risk approach to maximize returns over long term"))
                                    
        img = st.file_uploader("Upload your image")

        authenticator.logout("Logout", "sidebar")
except Exception as e:
    st.error(e)


    
