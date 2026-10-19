from pathlib import Path
from typing import Union

import logging
import json

logger = logging.getLogger(__name__)

def read_schema(schema: Union[str, Path]) -> dict:
    structure_path = Path(schema)

    try:
        with open(structure_path, "r", encoding='utf-8') as structure_file:
            structure_schema = json.load(structure_file)

    except FileNotFoundError:
        logger.error(f"Struture schema file not found: {structure_path}")
        return {}

    except json.JSONDecodeError as exep:
        logger.error(f"Could not parse JSON file {structure_path}: {exep}")
        return {}

    except Exception as exep:
        logger.error(f"Unknown error occured: {exep}")
        return {}

    if not structure_schema:
        return {}

    return structure_schema