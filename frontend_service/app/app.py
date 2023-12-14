from flask import Flask, render_template, request
import requests
from typing import Any, Union
app = Flask(__name__)

svm_service_url = 'http://svm_service:6000'
vgg_service_url = 'http://vgg_service:7000'
@app.route('/')
def hello_world():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def classify():
      music_file = request.files['musicFile']
      file_path = '/Nouvarch/shared_volume/' + music_file.filename
      music_file.save(file_path)
      files = {'musicFile': (music_file.filename, open(file_path, 'rb'))}
      response = requests.post(f'{svm_service_url}/classify', files=files)
      response_data = response.json()
      received_message = response_data.get("received_message", "No message received")
      svm_response = response_data.get("response", "No response received")
      return render_template('upload.html',result=svm_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=True)
