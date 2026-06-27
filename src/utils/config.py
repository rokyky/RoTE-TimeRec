'''Configuration loading utilities.

Reference:
    - PyYAML + dict-based config with defaults
'''

import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(base_dir, 'configs', 'default.yaml')

    # Load defaults
    if os.path.exists(default_path):
        with open(default_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with user config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        config = _deep_merge(config, user_config)
    elif config_path:
        raise FileNotFoundError(f'Config not found: {config_path}')

    return config

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result