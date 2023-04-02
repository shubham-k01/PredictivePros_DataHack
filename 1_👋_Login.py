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

authenticator = stauth.Authenticate(credentials,
    "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main" )


st.session_state["risk"] = ""
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    st.title("👋 Personal Finance")

    name = st.text_input("Name:")
    no_of_deps = st.number_input("Number of dependents:",0,10,step=1)
    age = st.multiselect("What is your age?",("under 30 years","31 to 39 years","40 to 50 years","Over 50 years"))
    experience = st.multiselect("Investing Experience?",("I have never invested before","1 to 3 years","4 to 7 years","More than 7 years"))
    dependants = st.multiselect("Who depends on you finncially?",("Kid(s)","Spouse","Parents","Other family","No dependents"))
    risk_app = st.selectbox("How much of a temporary drop in portfolio value could you handle?",("I can't take any drop","A drop between 5-10%","A drop between 10-20%","Greater than 20%"))
    dependants = st.multiselect("How do you want to invest your money?",("Balanced: Focussed on higher returns while minimising risk",
                                                                        "Growth: Focussed on superior long-term returns with moderate risk",
                                                                        "Aggressive: High-risk approach to maximize returns over long term"))
    submitted = st.button("Submit")
    authenticator.logout("Logout", "sidebar")
    risk = ""
    if submitted:
        if (risk_app=="I can't take any drop"):
            risk = "0%"
        elif (risk_app == "A drop between 5-10%"):
            risk = "10%"
        elif (risk_app == "A drop between 10-20%"):
            risk = "20%"
        else:
            risk = "30%"

        st.session_state["risk"] = risk
        st.write("Please go to Dashboard tab to view the results")
    

    

