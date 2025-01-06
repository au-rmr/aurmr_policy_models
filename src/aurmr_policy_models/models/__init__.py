import importlib

# model_name is for example "diffusion_policy.DiffusionPolicyModel"
def get_model(model_name, config):
    """
    Get the model class from the given model name.
    
    Args:
        model_name (str): Name of the model class to retrieve.
    
    Returns:
        class: The model class corresponding to the given name.
    """
    # Import the model class from the models directory
    module_name, class_name = model_name.rsplit('.', 1)
    module = importlib.import_module('aurmr_lfd.models.' + module_name)
    cfg_class = getattr(module, class_name + 'Config')
    cfg = cfg_class(**config)
    model_class = getattr(module, class_name)
    return model_class(cfg)