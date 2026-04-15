import os
import logging
import json

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

def file_exists(file_path):
    is_exist = os.path.exists(file_path)
    if is_exist:
        logging.info(f"File exists: {file_path}")
        return True
    else:
        logging.info(f"File not found: {file_path}")
        return False

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        if yaml is not None:
            content = yaml.safe_load(yaml_file)
        else:
            content = json.load(yaml_file)
    logging.info(f"Loaded YAML: {path_to_yaml}")
    return content

def create_dirs(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Directory created: {dir}")