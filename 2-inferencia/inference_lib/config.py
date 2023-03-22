
from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent.parent / 'config.yaml'
with CONFIG_FILE.open() as f:
    config = yaml.safe_load(f)