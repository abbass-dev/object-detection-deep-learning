import json

def parse_configuration(config_file):
    with open(config_file) as config_file:
        return json.load(config_file)