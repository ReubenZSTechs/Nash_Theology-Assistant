from pathlib import Path
from typing import Union

import logging
import yaml


logger = logging.getLogger(__name__)

REQUIRED_KEYS = ["agent", "system_prompt", "generation"]

def read_config(config: Union[str, Path]) -> dict:
    agent_path = Path(config)

    try:
        with open(agent_path, "r", encoding='utf-8') as agent_file:
            agent_config = yaml.safe_load(agent_file)

    except FileNotFoundError:
        logger.error(f"Agent configuration file not found: {agent_path}")
        return {}

    except yaml.YAMLError as exep:
        logger.error(f"Could not parse YAML file {agent_path}: {exep}")
        return {}

    except Exception as exep:
        logger.error(f"Unknown error occured: {exep}")
        return {}

    if agent_config is None:
        return {}

    # Validate missing keys:
    missing = [field for field in REQUIRED_KEYS if field not in agent_config]
    if missing:
        raise KeyError(f"{agent_path} is missing required key(s): {" ".join(missing)}")

    return agent_config