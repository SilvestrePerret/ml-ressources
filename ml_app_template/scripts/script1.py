# external imports
import click
import dotenv

dotenv.load_dotenv()  # noqa

# internal imports
from src.ml_engine import MLEngine


@click.command()
def predict():
    MLEngine.predict()


if __name__ == "__main__":
    predict()