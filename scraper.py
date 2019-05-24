

def scrape_web(stock):
    import datetime as dt
    import matplotlib.pyplot as plt
    from matplotlib import style
    import pandas as pd
    import pandas_datareader.data as web
    style.use('ggplot')
    from datetime import datetime
    
    
    start = dt.datetime(2015,1,1)
    end = datetime.today()
    
    
    df = web.DataReader(stock, 'iex', start, end) 
    
    df.reset_index(inplace=True)
    df.rename(index=str, columns={'close_price': 'Close', 
                                 'high_price': 'High',
                                 'low_price': 'Low',
                                 'open_price': 'Open',
                                 'volume': 'Volume',
                                 'begins_at': 'Date'}, inplace=True)
    df.set_index('date', inplace=True)
    df.head()
    import os
    os.remove("stocks.csv")

    df.to_csv('stocks.csv')

