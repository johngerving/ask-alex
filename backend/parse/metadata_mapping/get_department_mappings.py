import os
from typing import Dict
import json

MAPPINGS_FILENAME = os.path.join(os.path.dirname(__file__), "department_mappings.json")


def get_department_mappings() -> Dict[str, str]:
    """Retrieves a mapping of variant department names to canonicalized department names."""

    with open(MAPPINGS_FILENAME, "r") as f:
        data = json.load(f)

    mappings: Dict[str, str] = {}

    for el in data:
        if "variations" not in el or "canonical" not in el:
            raise Exception(
                f"Invalid mapping element {el}. Must contain 'variations' and 'canonical' keys."
            )

        if not isinstance(el["variations"], list) or not isinstance(
            el["canonical"], str
        ):
            raise Exception(
                f"Invalid mapping element {el}. 'variations' must be a list and 'canonical' must be a string."
            )

        for variation in el["variations"]:
            mappings[variation] = el["canonical"]

    return mappings
