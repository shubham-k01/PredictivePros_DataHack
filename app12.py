import toml 
import streamlit as st 
import pandas as pd
import snowflake.connector as sf
from datetime import date 

sidebar=st.sidebar

def fetch_pandas_old(cur, sql):
    cur.execute(sql)
    rows = 0
    while True:
        dat = cur.fetchmany(50000)
        if not dat:
            break
        df = pd.DataFrame(dat, columns=cur.description)
        rows += df.shape[0]
    print(rows)

def connect_to_snowflake(acct,usr,pwd,rl,wh,db):
    ctx=sf.connect(user=usr,account=acct,password=pwd,role=rl,warehouse=wh,database=db)
    cs=ctx.cursor()
    st.session_state["snow_conn"]=cs
    st.session_state["is_ready"]=True
    return cs

# @st.cache(supress_st_warnings=True,show_spinner=False)
def get_data():
    query='SELECT * FROM BANKING_DATA_ATLAS.BANKING.AEBIFBD2018'
    results=st.session_state["snow_conn"].execute(query)
    results=st.session_state["snow_conn"].fetch_pandas_all()
    return results



with sidebar:
    account=st.text_input("Account")
    username=st.text_input("Username")
    password=st.text_input("Password",type="password")
    role=st.text_input("Role")
    wh=st.text_input("Warehouse")
    db=st.text_input("Database")
    connect=st.button("Connect to snowflake",\
                       on_click=connect_to_snowflake,\
                        args=[account,username,password,role,wh,db])
    
if  'is_ready' not in st.session_state:
    st.session_state["is_ready"]=False
if st.session_state["is_ready"]==True:
    # st.write("Conneccted")
    data=get_data()
    # print("Banking Financial")
    st.title("Banking Financial Report")
    balances=data["Value"].agg(["min","max"])
    # print(balances["min"])
    min,max=st.slider("Value Range",
        min_value=float(balances["min"]),
        max_value=float(balances["max"]),
        value=[float(balances["min"]),float(balances["max"])]
                      )
    
    # data.loc[data["Value"].between(min,max)]
    df=st.dataframe(data)
    st.header("Bar Chart")
    st.bar_chart(data=df)

    # data[data[]]
    # print(type(data['Value'][0]))
    # st.dataframe(data)
