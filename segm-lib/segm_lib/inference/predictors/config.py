import importlib.resources as pkg_resources

import yaml

CONFIG_FILE = pkg_resources.files(__package__).joinpath('config.yaml')
with CONFIG_FILE.open('r') as f:
	config = yaml.safe_load(f)