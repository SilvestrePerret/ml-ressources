# external imports
from dotenv import load_dotenv

load_dotenv()  # noqa

# internal imports
from app import create_app


app = create_app("production")

if __name__ == "__main__":
    app.run()
