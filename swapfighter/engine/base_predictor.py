import torch

class BasePredictor:
    def __init__(self, cfg_dict, weights=None):
        from ..cfg import Config
        task = cfg_dict.pop('task', 'detect')
        self.cfg = Config.from_kwargs(task, **cfg_dict)
        self.weights = weights
        self.output_image = cfg_dict.get('output_image', None)
        self.output_txt = cfg_dict.get('output_txt', None)
        self.output_json = cfg_dict.get('output_json', None)
        self.class_names = cfg_dict.get('class_names', None)

        device_pref = cfg_dict.get('device', 'cuda')
        if device_pref == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = None

    def build_model(self):
        raise NotImplementedError

    def run(self, source):
        raise NotImplementedError