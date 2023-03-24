
from pathlib import Path
import __main__

import yaml

CONFIG_FILE = Path(__main__.__file__).parent / 'config.yaml'
with CONFIG_FILE.open() as f:
    config = yaml.safe_load(f)