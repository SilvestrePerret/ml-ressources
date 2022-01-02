# external imports
from flask import Flask

# internal imports
from app.config import CONFIG


def create_app(config_name):
    """App factory (useful to have tests !)"""
    app = Flask(__name__)
    app.config.from_object(CONFIG[config_name])

    # import routes and co. without using flask.blueprint
    with app.app_context():
        from . import routes  # noqa
        from . import errors  # noqa
    return app