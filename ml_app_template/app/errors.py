# external imports
from flask import current_app as app
import marshmallow

# internal imports
# from src.exceptions import ...


@app.errorhandler(marshmallow.exceptions.ValidationError)
def handle_validation_error(error):
    return error.messages, 400
