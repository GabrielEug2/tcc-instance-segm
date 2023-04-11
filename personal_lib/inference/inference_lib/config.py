import importlib.resources as pkg_resources

import yaml

CONFIG_FILE = pkg_resources.path(__package__, 'config.yaml')
with CONFIG_FILE.open() as f:
	config = yaml.safe_load(f)
	
def update_config(user_config):
	config['yolact']['dir'] = user_config['yolact_dir']
	config['solo']['dir'] = user_config['solo_dir']