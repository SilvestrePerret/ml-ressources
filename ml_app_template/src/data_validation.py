# standard imports
from typing import Dict
from collections import defaultdict

# external imports
import pandera as pa


class StrictSchema(pa.DataFrameSchema):
    def __init__(self, *args, **kwargs):
        kwargs["strict"] = True
        super().__init__(*args, **kwargs)

    def validate(self, *args, **kwargs):
        kwargs["lazy"] = True
        return super().validate(*args, **kwargs)


ExampleSchema = StrictSchema(
    columns={
        "a": pa.Column(str),
        "b": pa.Column(str),
        "ts": pa.Column("datetime64[ns, UTC]", coerce=True, nullable=True),
        "c": pa.Column(str),
    },
    checks=pa.Check(lambda df: not df.empty),
)


Example2Schema = ExampleSchema.set_index(["objectif"])

def standardize_error_messages(error: pa.errors.SchemaErrors) -> Dict:
    messages = defaultdict(list)
    for _, row in error.failure_cases.iterrows():
        if row["check"] == "column_in_dataframe":
            messages["missing_column"].append(
                f"Input data must have a '{row['failure_case']}' column."
            )
        elif row["check"] == "column_in_schema":
            messages["unkown_column"].append(f"Unknown column: '{row['failure_case']}'.")
        elif row["column"] is not None:
            if "dtype" in row["check"]:
                messages[row["column"]].append(
                    f"Invalid type (expected: '{row['check'][7:-2]}', actual: '{row['failure_case']}')."
                )
            elif row["check"] == "not_nullable":
                messages[row["column"]].append("Nan values are not allowed.")
        else:
            messages["other"].append(str(row.to_dict()))
    return dict(messages)
