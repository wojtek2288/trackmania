import json
import torch

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        else:
            return super(JSONEncoder, self).default(obj)

class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.convert_to_tensor, *args, **kwargs)

    def convert_to_tensor(self, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, list):
                data_dict[key] = torch.tensor(value)
        return data_dict

