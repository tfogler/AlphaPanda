
_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls
    return decorator


def get_model(cfg, device):
    model = _MODEL_DICT[cfg.type](cfg, device=device)
    print("Get model before to(device): {}".format(next(model.parameters()).device))
    return model
