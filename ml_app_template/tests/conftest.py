# external imports
import pytest

@pytest.fixture(scope="session")
def my_fixture():
    return "my_fixture"
