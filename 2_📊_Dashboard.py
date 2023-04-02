import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import database as db


import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import date,timedelta
import seaborn as sns
import matplotlib.pyplot as plt
# from ..login import authenticator

risk = st.session_state["risk"] 

@st.cache_data
def dashboard(risk):
    assets1=["HCLTECH.NS","TRENT.NS","TATAMOTORS.NS","BAJAJFINSV.NS","HDFCBANK.NS","DRREDDY.NS"]
    assets2=["TCS.NS","RELIANCE.NS","ITC.NS","TATASTEEL.NS","ICICIBANK.NS","PIDILITIND.NS"]
    assets3=["INFOSYS.NS","HDFCBANK.NS","VEDL.NS","ASIANPAINT.NS","M&M.NS"]
    assets4=["WIPRO.NS","MARUTI.NS","TITAN.NS","HINDALCO.NS","GODREJCP.NS"]
    if(risk=="10%"):
        assets=assets1
    elif(risk=="20%"):
        assets=assets2
    elif(risk=="30%"):
        assets=assets3
    else:
        assets=assets4
    end = date.today()
    start = end - timedelta(days=2000)
    assets.sort()

    # Downloading data
    data = yf.download(assets, start = start, end = end)
    data = data.loc[:,('Adj Close', slice(None))]
    data.columns = assets

    corr = data.corr()
    # mask = np.zeros_like(corr)
    # mask[np.triu_indices_from(mask)] = True

    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    mu = mean_historical_return(data)
    S = CovarianceShrinkage(data).ledoit_wolf()

    from pypfopt.efficient_frontier import EfficientFrontier

    ef = EfficientFrontier(mu, S)
    ef.efficient_risk(target_volatility=0.21)
    weights = ef.clean_weights()
    import plotly.graph_objects as go
    labels=assets
    values=pd.Series(weights)
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
    insidetextorientation='radial')])
    # sns.heatmap(corr, vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
    # plt.yticks(rotation=0) 
    # plt.xticks(rotation=90)
    # plt.title("Portfolio Correlation") 
    # plt.show()
    import plotly.express as px

    # Streamlit
    st.title("📊 DashBoard")
    fig1 = px.imshow(corr)
    st.plotly_chart(fig1)
    st.write(weights)

    w=ef.portfolio_performance(verbose=True)

    # authenticator.logout("Logout", "sidebar")
    
    st.write(f"Expected Annual return :{w[0]}")
    st.write(f"Annual volatility: {w[1]}")
    st.write(f"Sharpe Ratio: {w[2]}")
    st.plotly_chart(fig)
    
if "risk" not in st.session_state:
    st.session_state["risk"] = None
    st.write("An error occured because there is no risk")
else:
    dashboard(risk)

