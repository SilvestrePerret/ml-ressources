# external imports
from werkzeug.datastructures import ImmutableMultiDict
from marshmallow import Schema, fields, validate, pre_load


class A(Schema):
    a = fields.Url(required=True, data_key="alpha")

    @pre_load
    def extract_args(self, data, **kwargs):
        if isinstance(data, ImmutableMultiDict):
            args_dict = data.to_dict(flat=False)
            for name, field in self.declared_fields.items():
                if field.data_key is not None:
                    name = field.data_key
                if not isinstance(field, fields.List) and name in args_dict:
                    args_dict[name] = args_dict[name][0]
            return args_dict
        return data


class B(Schema):
    flags = fields.Dict(keys=fields.Str(), values=fields.Bool(), required=True)


a_schema = A()
b_schema = B()
