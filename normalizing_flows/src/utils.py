import yaml


def load_config(config_path):
    """
	Args:
		config_path (str):

	Returns:
		dict
	"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
