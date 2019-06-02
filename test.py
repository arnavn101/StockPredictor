from flask import Flask, render_template, request
from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()
app = Flask(__name__)

@app.route('/')
def data():
   return render_template('test.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      
      name = request.form.get("Name")
      days = request.form.get("Days")
      
      from scraper import scrape_web
      scrape_web(name)
      
      
      from rnn_robust import rnn_predict
      code = rnn_predict(name, days)
      return code
      """
    code + render_template('results.html')
      """
      
if __name__ == '__main__':
   app.run(host='127.0.0.1',port=12345)
