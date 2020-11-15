from src.etl import complete
import json
import sys

def main():
    if targets[0] == "cora":
        with open('config/cora_params.json') as fh:
            data_cfg = json.load(fh)
    if targets[0] == "twitch":
        with open('config/twitch_params.json') as fh:
            data_cfg = json.load(fh)
    else:
        with open('config/cora_params.json') as fh:
            data_cfg = json.load(fh)
    complete(**data_cfg)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main()
