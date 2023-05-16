import sys,os
from flask import Flask
from housing.logger import logging
from housing.exception import Exception_Handling
from housing.constant import CONFIG_DIR, get_current_time_stamp
from housing.pipeline.pipeline import Pipeline
from housing.entity.housing_predictor import HousingPredictor, HousingData
from housing.logger import get_log_dataframe
from flask import send_file, abort, render_template
from housing.logger import get_log_dataframe
from housing.config.configuration import Configuration
from housing.util.util import read_yaml_file, write_yaml_file

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = 'logs'
PIPELINE_FOLDER_NAME = "housing"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path .join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

app = Flask(__name__)

@app.route('/artifact', defaults={'req_path': 'housing'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("housing", exist_ok=True)
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # if path doesn't exist, return 404
    if not os.path.exists(abs_path):
        return abort(404)
    
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)
    
    # Displaying directory contents
    files = {
        os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if "artifact" in os.path.join(abs_path, file_name)
    }
    result = {
        "files":files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)

@app.route("/", methods=['GET','POST'])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/view_experiment_history', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes="table table-striped col-12")
    }
    return render_template("experiment_history.html", context=context)

@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started"
        pipeline.start()
    else:
        message = "Training already in progress"
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes="table table-striped col-12"),
        "message": message
    }
    return render_template("train.html", context=context)


if __name__=="__main__":
    app.run(debug=True)