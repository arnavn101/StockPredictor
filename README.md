# StockPredictor

My  program is a stock predictor. When it is run, the program will automatically open a html page that prompts the user for the number of days from today and the name of the stock to predict its price. Then, the program will access yahoo finance and use previous stock prices from last 5 years in order to form an accurate model. When this finishes, an html page with profit or loss of stock opens. Additionally, an interactive graph also opens, portraying daily future predictions. This program takes about 3 minutes to run, excluding the requirement for users to interact with Python code and its sophisticated elements.

* It makes use of Keras, Sklearn modules and Flask integration to accomplish its task

* This program can currently be run from only any interface with python installed.  

Steps for Running the application:

1) pip install -r requirements.txt

2) python flaskapp.py
