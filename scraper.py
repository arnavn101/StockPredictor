def scrape_web(stock):
    
    import datetime as dt
    from matplotlib import style
    import pandas_datareader.data as web
    from pandas_datareader import data as pdr
    style.use('ggplot')
    from datetime import datetime
    from iexfinance.stocks import get_historical_data
    import yfinance as yf
    
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
