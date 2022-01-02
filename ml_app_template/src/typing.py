# external imports
from pandera.typing import DataFrame

# internal imports
from src.data_validation import (
    Example2Schema,
)

Example2DataFrame = DataFrame[Example2Schema]