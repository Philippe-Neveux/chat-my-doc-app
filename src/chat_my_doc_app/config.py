"""
Configuration management for Chat My Doc App.

This module handles loading configuration from YAML files and environment variables.
Sensitive data (API keys, hosts) come from environment variables, while
non-sensitive settings come from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.

    Returns:
        Dictionary containing configuration data

    Raises:
        FileNotFoundError: If config file is not found
        yaml.YAMLError: If YAML parsing fails
    """
    if config_path is None:
        config_file_path = Path(__file__).parent / "config" / "config.yaml"
    else:
        config_file_path = Path(config_path)

    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    try:
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        logger.info(f"Loaded configuration from: {config_file_path}")
        return config_data or {}

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise
