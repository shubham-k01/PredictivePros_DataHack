from matplotlib import markers
from nbformat import write
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pandas_datareader as data
import datetime
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
import calendar 
import database_expense as db_exp
from datetime import date,timedelta
import yfinance as yf
import toml 
from datetime import date 
import plotly.express as px
from yahooquery import Ticker
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import database as db
# import snowflake.connector as sf
from plotly import graph_objs as go
from streamlit_folium import folium_static
import folium
import matplotlib as plt
import streamlit.components.v1 as components
import openai
import streamlit as st
from streamlit_chat import message
import os 
from dotenv import load_dotenv
load_dotenv('api_key.env')
openai.api_key = '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'
# openai.api_key = os.environ.get('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')
def generate_response(prompt):
    completion=openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.6,
    )
    message=completion.choices[0].text
    return message

st.set_page_config(
   page_title="Consultancy Portal",
   page_icon="./logo.png",
   layout="wide",
   initial_sidebar_state="expanded",
)


st.sidebar.image("./static/images/logo1.png")
# st.sidebar.image("./static/images/my_logo.png")
st.sidebar.title('Portal Menu')
rad1 =st.sidebar.radio("Navigation",["Stocks Centre","Analysis","Personal Finance","Personal Finance Dashboard", "Expense Tracker","Insights","Virtual Assistant", "About-Us"])

if rad1 == "Stocks Centre": 

    # START = (datetime.date.today() - datetime.timedelta(365))
    # TODAY = datetime.date.today() + datetime.timedelta(7)


    model = load_model('./Future Stock Prediction Using Last 7 Days Moving Averages.h5 ')

    st.title('Stock Portal')
    st.subheader("NIFTY-50")
    nifty_df = pd.read_csv('^NSEI.csv')
    
    def plot_nifty_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty_df['Date'], y=nifty_df['Close'], name='stock_close'))
        fig.layout.update(title_text="NIFTY_50", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig,  use_container_width = True)
    

    plot_nifty_data()
    st.experimental_data_editor(nifty_df, use_container_width=True)

    st.subheader("Stock price overview")
    # selected_stock = st.text_input('Enter stock ticker (add .NS)')
    stocks = ('RELIANCE.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'HCLTECH.NS', 'ITC.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'HINDUNILVR.NS', 'LT.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS', 'TECHM.NS', 'TATACONSUM.NS', 'AXISBANK.NS', 'HEROMOTOCO.NS','BAJAJFINSV.NS', 'DRREDDY.NS', 'JSWSTEEL.NS', 'CIPLA.NS', 'UPL.NS', 'POWERGRID.NS', 'SBIN.NS', 'BPCL.NS', 'M&M.NS', 'HINDUNILVR.NS', 'INDUSINDBK', 'ONGC.NS', 'BRITANNIA.NS', 'HINDALCO.NS', 'HDFCLIFE.NS', 'BHARTIARTL.NS', 'NTPC.NS', 'TATASTEEL.NS', 'EICHERMOT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'COALINDIA.NS', 'SBILIFE.NS', 'ADANIENT.NS', 'MARUTI.NS', 'TITAN.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'ADANIPORTS.NS', 'SUNPHARMA.NS', 'APOLLOHOSP.NS' )
    selected_stock = st.selectbox('Top 10 NIFTY-50 stocks', stocks)
    
    # n_years = st.slider('Years of historical data:', 1, 10)
    # period = n_years * 365

    intervals = ('Daily', 'Weekly', 'Monthly', 'Quaterly')
    input = st.selectbox('Interval', intervals)

    if input == 'Daily':
        interval = '1d'
    elif input == 'Weekly':
        interval = '1wk'
    elif input == 'Monthly':
        interval = '1mo'
    elif input == 'Quaterly':
        interval = '3mo'
    else:
        interval = '1d'

    input_months = 0
    # input_months = st.number_input('Enter No of previous Months Historical Data', min_value = 0,max_value=12,value = 0,step = 1)


    if input_months == 0 :
        START = (datetime.date.today() - datetime.timedelta(365))
    else:
        START = datetime.date.today() - datetime.timedelta(30*(input_months))
    TODAY = datetime.date.today() + datetime.timedelta(7)

    @st.cache_resource
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY, interval= interval)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load data....")
    data = load_data(selected_stock)
    # data_load_state.text("Loading data....Done!")
    
    st.subheader('Raw Data')

    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text=selected_stock, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig,  use_container_width = True)

    plot_raw_data()

    df = data.copy()

    df.reset_index(inplace=True)

    x_future = np.array(df.Close.rolling(7).mean()[-7:])

    scale = max(x_future) - min(x_future)
    minimum = min(x_future)

    for i in range(0, len(x_future)):
        x_future[i] = (x_future[i] - minimum) / scale

    x_future = np.reshape(x_future, (1, 7, 1))

    y_future = []

    while len(y_future) < 7:
    #     Predicting future values using 7-day moving averages of the last day 7 days.
        p = model.predict(x_future)[0]
        
    #     Appending the predicted value to y_future
        y_future.append(p)
        
    #     Updating input variable, x_future
        x_future = np.roll(x_future, -1)
        x_future[-1] = p

    y_future = np.array(y_future)
    y_future = np.reshape(y_future, (7))

    for i in range(0, len(y_future)):
        y_future[i] = (y_future[i] * scale) + minimum

    y_future = np.reshape(y_future, (7, 1))

    last7 = pd.DataFrame(df.Close[-7:])
    last7.reset_index(drop=True, inplace=True)
    y_future = pd.DataFrame(y_future, columns=['Close'])
    predictions = pd.concat([last7, y_future], ignore_index=True)

    prev_7 = datetime.date.today() - datetime.timedelta(7)
    predictions['Date'] = [prev_7 + datetime.timedelta(x) for x in range(0, 14)]

    st.subheader('Predicted Data')

    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(predictions[8:13])
    


    def plot_predicted_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'][-7:], y=data['Close'][-7:], name='current_stock_price'))
        
        fig.add_trace(go.Scatter(x=predictions['Date'][8:13], y=predictions['Close'][7:], name='predicted_stock_price'))
        fig.add_trace(go.Scatter(x=predictions['Date'][7:9], y=predictions['Close'][6:8], name = '--', mode='lines', line=dict(color='royalblue', width=4, dash='dot')))
        fig.layout.update(title_text = selected_stock, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig,  use_container_width = True)

    plot_predicted_data()

    st.subheader("Begin you investing journey with us.....")
    st.write("You can buy or sell stocks via thir party plugins, we partner with all the top brokers so that you can invest comfortably.")
    with st.container():
        col1, col2 = st.columns((1,2))
        with col1:
            if st.button("Buy"):
                st.markdown("[To buy stocks: ](https://www.gateway-tt.in/trade?orderConfig=%5B%7B%22type%22%3A%22buy%22%2C%22quantity%22%3A10%2C%22ticker%22%3A%22RELIANCE%22%7D%5D&cardsize=big&withSearch=true&withTT=true)")
        with col2:
            if st.button("Sell"):
                st.markdown("[To sell stocks: ](https://www.gateway-tt.in/trade?orderConfig=%5B%7B%22type%22%3A%22sell%22%2C%22quantity%22%3A10%2C%22ticker%22%3A%22RELIANCE%22%7D%5D&cardsize=big&withSearch=true&withTT=true)")

    # st.title("ChatGPT-like Web App")
    # #storing the chat
    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = []
    # if 'past' not in st.session_state:
    #     st.session_state['past'] = []
    # user_input=st.text_input("You:",key='input')
    # if user_input:
    #     output=generate_response(user_input)
    #     #store the output
    #     st.session_state['past'].append(user_input)
    #     st.session_state['generated'].append(output)
    # if st.session_state['generated']:
    #     for i in range(len(st.session_state['generated'])-1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# if rad1 == "Buy N Sell":
#     components.html(<p class="sc-embed" data-width="500px" data-orders="%5B%7B%22type%22%3A%22buy%22%2C%22quantity%22%3A10%2C%22ticker%22%3A%22RELIANCE%22%7D%5D" data-cardsize="big" data-withtt="true" data-withsearch="true" style="width:500px;min-height:300px;display:flex;align-items:center;justify-content:center"> <strong>loading widget to trade RELIANCE</strong> </p> <script async src="https://www.gateway-tt.in/assets/embed.js"></script> )

if rad1 == "Analysis":
    st.title("Stock Analyser")
    st.subheader("Providing you with comprehensive reporting of all the important pointers and KPI's you need to know before investing.")
    stocks = ('RELIANCE.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'HCLTECH.NS', 'ITC.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'HINDUNILVR.NS', 'LT.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS', 'TECHM.NS', 'TATACONSUM.NS', 'AXISBANK.NS', 'HEROMOTOCO.NS','BAJAJFINSV.NS', 'DRREDDY.NS', 'JSWSTEEL.NS', 'CIPLA.NS', 'UPL.NS', 'POWERGRID.NS', 'SBIN.NS', 'BPCL.NS', 'M&M.NS', 'HINDUNILVR.NS', 'INDUSINDBK', 'ONGC.NS', 'BRITANNIA.NS', 'HINDALCO.NS', 'HDFCLIFE.NS', 'BHARTIARTL.NS', 'NTPC.NS', 'TATASTEEL.NS', 'EICHERMOT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'COALINDIA.NS', 'SBILIFE.NS', 'ADANIENT.NS', 'MARUTI.NS', 'TITAN.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'ADANIPORTS.NS', 'SUNPHARMA.NS', 'APOLLOHOSP.NS' )
    selected_stock1 = st.selectbox('NIFTY-50 stocks', stocks)
    stock = Ticker(selected_stock1)
    # stock.company_officers
    st.write("Earnings history of the stock is: ")
    stock.earning_history
    stock.balance_sheet()
    stock.cash_flow(trailing=False)
    stock.income_statement(frequency='q')
    st.write("Valuation measures of the company are: ")
    stock.valuation_measures
    stock.all_financial_data('q')
    types = ['TotalDebt', 'TotalAssets', 'EBIT', 'EBITDA', 'PeRatio']
    stock.get_financial_data(types, trailing=False)
    st.write("Summary of all important terms related to better analysing the comapny are: ")
    stock.summary_detail


if rad1 == "Personal Finance":
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
    
if rad1 == "Personal Finance Dashboard":
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

if rad1 == "Expense Tracker":
    # -------------- SETTINGS --------------
    incomes = ["Salary", "Blog", "Other Income"]
    expenses = ["Rent", "Utilities", "Groceries", "Car", "Other Expenses", "Saving"]
    # forecast_savings=["Anual Salary Growth","Annual Inflation Rate","Forecast Years"]
    currency = "INR"
    page_title = "Income and Expense Tracker"
    page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    layout = "centered"
    # --------------------------------------

    # st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    st.title(page_title + " " + page_icon)

    # --- DROP DOWN VALUES FOR SELECTING THE PERIOD ---
    years = [date.today().year, date.today().year + 1]
    months = list(calendar.month_name[1:])


    # --- DATABASE INTERFACE ---
    def get_all_periods():
        items = db_exp.fetch_all_periods()
        periods = [item["key"] for item in items]
        return periods


    # --- HIDE STREAMLIT STYLE ---
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # --- NAVIGATION MENU ---
    selected = option_menu(
        menu_title=None,
        options=["Data Entry", "Data Visualization"],
        icons=["pencil-fill", "bar-chart-fill"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
    )

    # --- INPUT & SAVE PERIODS ---
    if selected == "Data Entry":
        st.header(f"Data Entry in {currency}")
        with st.form("entry_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            col1.selectbox("Select Month:", months, key="month")
            col2.selectbox("Select Year:", years, key="year")

            "---"
            with st.expander("Income"):
                for income in incomes:
                    st.number_input(f"{income}:", min_value=0, format="%i", step=10, key=income)
            with st.expander("Expenses"):
                for expense in expenses:
                    st.number_input(f"{expense}:", min_value=0, format="%i", step=10, key=expense)
            with st.expander("Comment"):
                comment = st.text_area("", placeholder="Enter a comment here ...")

            "---"
            # with st.expander("Forecast Savings: "):
            #     for forecast in forecast_savings:
            #         st.number_input(f"{forecast}:",min_value=0,format="%i",step=1,key=forecast)

            submitted = st.form_submit_button("Save Data")
            if submitted:
                period = str(st.session_state["year"]) + "_" + str(st.session_state["month"])
                incomes = {income: st.session_state[income] for income in incomes}
                expenses = {expense: st.session_state[expense] for expense in expenses}
                db.insert_period(period, incomes, expenses, comment)
                st.success("Data saved!")


    # --- PLOT PERIODS ---
    if selected == "Data Visualization":
        st.header("Data Visualization")
        with st.form("saved_periods"):
            period = st.selectbox("Select Period:", get_all_periods())
            inflation = st.slider("Inflation Rate",0,15)
            growth = st.slider("Annual Salary Growth",0,15)
            submitted = st.form_submit_button("Plot Period")
            
            if submitted:
                # Get data from database
                period_data = db_exp.get_period(period)
                comment = period_data.get("comment")
                expenses = period_data.get("expenses")
                incomes = period_data.get("incomes")

                # Create metrics
                total_income = sum(incomes.values())
                total_expense = sum(expenses.values())
                remaining_budget = total_income - total_expense
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Income", f"{total_income} {currency}")
                col2.metric("Total Expense", f"{total_expense} {currency}")
                col3.metric("Remaining Budget", f"{remaining_budget} {currency}")
                st.text(f"Comment: {comment}")
                monthly_inflation = (1+inflation)**(1/12) - 1
                cumulative_inflation_forecast = np.cumprod(np.repeat(1 + monthly_inflation, 12))
                forecast_expenses = total_expense*cumulative_inflation_forecast
                monthly_growth = (1+growth)**(1/12) - 1
                cumulative_growth_forecast = np.cumprod(np.repeat(1 + monthly_growth, 12))
                forecast_income = total_income*cumulative_growth_forecast
                forecast_budget = forecast_income-forecast_expenses

                # Create sankey chart
                label = list(incomes.keys()) + ["Total Income"] + list(expenses.keys())
                source = list(range(len(incomes))) + [len(incomes)] * len(expenses)
                target = [len(incomes)] * len(incomes) + [label.index(expense) for expense in expenses.keys()]
                value = list(incomes.values()) + list(expenses.values())

                # Data to dict, dict to sankey
                link = dict(source=source, target=target, value=value)
                node = dict(label=label, pad=20, thickness=30, color="#E694FF")
                data = go.Sankey(link=link, node=node)

                #Forecast Computation

                # Plot it!
                fig1 = go.Figure(data)
                fig1.update_layout(margin=dict(l=0, r=0, t=5, b=5))
                st.plotly_chart(fig1, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(
                go.Scatter(
                x=[0,1,2], 
                y=forecast_income,
                name="Forecast Income"
                )
            )

                fig.add_trace(
                        go.Scatter(
                            x=[0,1,2],
                            y=forecast_expenses,
                            name= "Forecast Expenses"
                        )
                    )

                fig.add_trace(
                        go.Scatter(
                                x=[0,1,2], 
                                y=forecast_budget   ,
                                name= "Forecast Savings"
                            )
                    )
                fig.update_layout(title='Forecast Salary, Expenses & Savings Over the Years',
                                xaxis_title='Year',
                                yaxis_title='Amount($)')

                st.plotly_chart(fig, use_container_width=True)

if rad1 == "Insights":
    st.title("Incredible Insights")

    st.markdown("## John Doe")
    st.caption("Writer and Data Analyst")
    st.markdown("This specially curated section, provides you with insightful news anout the latest developments in the financial world with helps of interactive visualisations and graphs from professional data analysts.")


    with st.container():
        image_col, text_col = st.columns((1,2))
        with image_col:
            st.image("./static/images/svb.jpeg")

        with text_col:
            st.subheader("Biggest bank failure in USA financial system!")
            st.write("""In the latest revelation, the favourite bank of startups collapsed strangling numerous startups in this already strained envoirment.
                """)
            st.markdown("[To know more click here...](https://docs.google.com/document/d/112V-VDBMwsmMM6hlOeTQSegG8YaQ7Mt7nF00zE3mg60/edit)")

    with st.container():
        image_col, text_col = st.columns((1,2))
        with image_col:
            st.image("./static/images/ccard.jpg")

        with text_col:
            st.subheader("Cred-visualizer")
            st.write("""In this latest segment, we take a deep dive into the spending habits of Indians by analysing credit card spending data to understand the main cash drain faced by an ordinary household.
                """)
            st.markdown("[To know more click here...](https://app.powerbi.com/view?r=eyJrIjoiOWExMDY3OTgtZjA5Ni00MTU5LThkZWEtZjUwMTM1OGZhMzMwIiwidCI6ImQxZjE0MzQ4LWYxYjUtNGEwOS1hYzk5LTdlYmYyMTNjYmM4MSIsImMiOjEwfQ%3D%3D)")

    with st.container():
        image_col, text_col = st.columns((1,2))
        with image_col:
            st.image("./static/images/covid.jpg")

        with text_col:
            st.subheader("Impact of COVID-19 on Indians")
            st.write("""A special insight on the devastating effects the COVID-19 pandemic had on the populus of India and how strict policies were enacted to curb and contain the spread of the deadly virus and to minimise it's impact.
                """)
            
            st.markdown("[To view detailed dashboard, click here!](https://app.powerbi.com/view?r=eyJrIjoiZTAwZDI0ZmItY2Q5Zi00YTllLThmM2UtYTYzMjFlMWIyNTk0IiwidCI6ImQxZjE0MzQ4LWYxYjUtNGEwOS1hYzk5LTdlYmYyMTNjYmM4MSIsImMiOjEwfQ%3D%3D)")

    with st.container():
        image_col, text_col = st.columns((1,2))
        with image_col:
            st.image("./static/images/ftx.jpg")

        with text_col:
            st.subheader("Trouble in Paradise: The FTX Collapse")
            st.write(""" The love of the crypto market, the apple of every crypto trader's eyes, FTX recently bought tears to those same eyes. Recent revelations of unethical accounting practices and embezellment led to the collapse of one of the biggest crypto exchanges in a matter of days!
                """)
            st.markdown("[Click here for the full story...](https://docs.google.com/document/d/14wwBJZGOl8a82aWNgbPePnSJ8PhgbdhSoyTJLaWNtEk/edit)")

# if rad1 == "Financial Analysis":
#     st.title("Finance for You!")
#     stocks = ('RELIANCE.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'HCLTECH.NS', 'ITC.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'HINDUNILVR.NS', 'LT.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS', 'TECHM.NS', 'TATACONSUM.NS', 'AXISBANK.NS', 'HEROMOTOCO.NS','BAJAJFINSV.NS', 'DRREDDY.NS', 'JSWSTEEL.NS', 'CIPLA.NS', 'UPL.NS', 'POWERGRID.NS', 'SBIN.NS', 'BPCL.NS', 'M&M.NS', 'HINDUNILVR.NS', 'INDUSINDBK', 'ONGC.NS', 'BRITANNIA.NS', 'HINDALCO.NS', 'HDFCLIFE.NS', 'BHARTIARTL.NS', 'NTPC.NS', 'TATASTEEL.NS', 'EICHERMOT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'COALINDIA.NS', 'SBILIFE.NS', 'ADANIENT.NS', 'MARUTI.NS', 'TITAN.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'ADANIPORTS.NS', 'SUNPHARMA.NS', 'APOLLOHOSP.NS' )
#     selected_stock1 = st.selectbox('Top 10 NIFTY-50 stocks', stocks)
#     msft = yf.Ticker(selected_stock1)
#     START = (datetime.date.today() - datetime.timedelta(365))
#     TODAY = datetime.date.today() + datetime.timedelta(7)


#     df = yf.download(selected_stock1, START, TODAY, interval= '1d')
#     st.write(df)
#     # st.write(msft.fast_info)
#     st.write(msft.get_shares_full(start="2022-01-01", end=None))
#     # st.write(msft.income_stmt)
#     # st.write(msft.quarterly_earnings)
#     st.write(msft.news)
#     st.write(msft.get_income_stmt())

if rad1 == "Virtual Assistant":
    st.title("Samagrh-bot")
    st.subheader("Hi, I am Samagrah-bot, your own virtual private assistant! Use me if you have any queries you want to have solved.")
    #storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    user_input=st.text_input("You:",key='input')
    if user_input:
        output=generate_response(user_input)
        #store the output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')    


# if rad1 == "Profile":
#     st.title("Your Profile")

#     col1 , col2 = st.columns(2)

#     rad2 =st.radio("Profile",["Sign-Up","Sign-In"])


#     if rad2 == "Sign-Up":

#         st.title("Registration Form")



#         col1 , col2 = st.columns(2)

#         fname = col1.text_input("First Name",value = "first name")

#         lname = col2.text_input("Second Name")

#         col3 , col4 = st.columns([3,1])

#         email = col3.text_input("Email ID")

#         phone = col4.text_input("Mob number")

#         col5 ,col6 ,col7  = st.columns(3)

#         username = col5.text_input("Username")

#         password =col6.text_input("Password", type = "password")

#         col7.text_input("Repeat Password" , type = "password")

#         but1,but2,but3 = st.columns([1,4,1])

#         agree  = but1.checkbox("I Agree")

#         if but3.button("Submit"):
#             if agree:  
#                 st.subheader("Additional Details")

#                 address = st.text_area("Tell Us Something About You")
#                 st.write(address)

#                 st.date_input("Enter your birth-date")

#                 v1 = st.radio("Gender",["Male","Female","Others"],index = 1)

#                 st.write(v1)

#                 st.slider("age",min_value = 18,max_value=60,value = 30,step = 2)

#                 img = st.file_uploader("Upload your profile picture")
#                 if img is not None:
#                     st.image(img)

#             else:
#                 st.warning("Please Check the T&C box")

#     if rad2 == "Sign-In":
#         col1 , col2 = st.columns(2)

#         username = col1.text_input("Username")

#         password =col2.text_input("Password", type = "password")

#         but1,but2,but3 = st.columns([1,4,1])

#         agree  = but1.checkbox("I Agree")

#         if but3.button("Submit"):
            
#             if agree:  
#                 st.subheader("Additional Details")

#                 address = st.text_area("Tell Us Something About You")
#                 st.write(address)

#                 st.date_input("Enter your birth-date")

#                 v1 = st.radio("Gender",["Male","Female","Others"],index = 1)

#                 st.write(v1)

#                 st.slider("age",min_value = 18,max_value=60,value = 30,step = 2)

#                 img = st.file_uploader("Upload your profile picture")
#                 if img is not None:
#                     st.image(img)
#             else:
#                 st.warning("Please Check the T&C box")

if rad1 == "About-Us": 
    st.title("Consultancy Portal")

    st.subheader("Mission")
    st.write("We are on a mission to educate the common retail investors about the intricracies of the seemingly complex world of financial analysis and to help them in their journey towards financial freedom. Our aim is to make investing and savings easy and accesible to each and every citizen of the country so that society flourishes ultimately making India a developed nation.")

    st.subheader("Team")
    # st.image(him.jpeg, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with st.container():
        image_col1, text_col = st.columns((1,2))
        with image_col1:
            st.image("./static/images/him.jpeg")

        with text_col:
            st.subheader("Himanshu Kakwani")
            st.caption("ML, Back-end")
            st.write("Himanshu Kakwani from Data Science branch of DJ Sanghvi College of Engineering is a smart and talented person who has contributed in building the project. He helped in building the various ML models used for financial analysis and also helped in integrating the same into the back-end of the web-app.")
            st.balloons()

    with st.container():
        image_col1, text_col = st.columns((1,2))
        with image_col1:
            st.image("./static/images/ankiy.jpg")

        with text_col:
            st.subheader("Ankit Ladva")
            st.caption("Database, ML")
            st.write("Ankit Ladva from Data Science branch of DJ Sanghvi College of Engineering is a smart and talented person who has contributed in building the project. He helped in building the various ML models used for financial analysis and also helped in integrating the same into the back-end of the web-app.")
    
    with st.container():
        image_col1, text_col = st.columns((1,2))
        with image_col1:
            st.image("./static/images/pr.jpg")

        with text_col:
            st.subheader("Pratham Goradia")
            st.caption("ML,Web design")
            st.write("Pratham Goradia from Data Science branch of DJ Sanghvi College of Engineering is a smart and talented person who has contributed in building the project. He helped in building the various ML models used for financial analysis and also helped in integrating the same into the back-end of the web-app.")

    with st.container():
        image_col1, text_col = st.columns((1,2))
        with image_col1:
            st.image("./static/images/photo.jpg")

        with text_col:
            st.subheader("Shubham Kurunkar")
            st.caption("Back-end, Web  Design")
            st.write("Shubham Kurunkar from Data Science branch of DJ Sanghvi College of Engineering is a smart and talented person who has contributed in building the project. He helped in building the various ML models used for financial analysis and also helped in integrating the same into the back-end of the web-app.")
    
    st.subheader('Locate Us')
    m = folium.Map(location=[19.107340, 72.837125], zoom_start=16, )

    # add marker for Bombay Stock Exhcange
    tooltip = "DJ Sanghvi"
    folium.Marker(
        [19.107340, 72.837125], popup="Our Headquarters", tooltip=tooltip
    ).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)

