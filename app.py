import sys
from flask import Flask
from housing.logger import logging
from housing.exception import Exception_Handling

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    try:
        raise Exception("Testing Custom exception")
    except Exception as e:
        housing_exception = Exception_Handling(e, sys)
        logging.info(housing_exception.error_message)
        logging.info("Testing the logging Module")
    return "House Price Prediction Project"

if __name__=="__main__":
    app.run(debug=True)