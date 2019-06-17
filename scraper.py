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
    #web.api_key("sk_572cad65deef4830a0042e66df919dbc")
    df = web.DataReader(stock, start, end) 
    #df = get_historical_data(stock, start, end)
    #df = yf.download(stock, start, end)
    
    
    # df.reset_index(inplace=True)
    # df.rename(index=str, columns={'close_price': 'Close', 
    #                              'high_price': 'High',
    #                              'low_price': 'Low',
    #                              'open_price': 'Open',
    #                              'volume': 'Volume',
    #                              'begins_at': 'Date'}, inplace=True)
    # df.set_index('date', inplace=True)
    # df.head()
    try:
        import os
        os.remove("stocks.csv")
    except Exception:
        pass

    df.to_csv('stocks.csv')
