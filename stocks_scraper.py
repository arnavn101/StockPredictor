import datetime as dt
from matplotlib import style
import pandas_datareader.data as web
from datetime import datetime
import yfinance as yf

def scrape_web(stock):
    style.use('ggplot')
    start = dt.datetime(2015,1,1)
    end = datetime.today()
    yf.pdr_override()
    df = web.DataReader(stock, start, end) 
    
    try:
        import os
        os.remove("stocks.csv")
    except Exception:
        pass

    df.to_csv('stocks.csv')
