# standard imports
import subprocess
import datetime

# external imports
from flask import current_app as app, request, jsonify

# internal imports
from app.schemas import a_schema, b_schema
from src.ml_engine import MLEngine


@app.route("/")
@app.route("/index")
def index():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Hello World ! (local time: {now})"


@app.route("/recommendation", methods=["GET"])
def prediction():
    parameters = a_schema.load(request.args)
    # build response
    predictions = MLEngine(parameters).get_predictions()

    response = {
        "request": {**parameters},
        "predictions": predictions,
    }
    return jsonify(b_schema.dump(response))
