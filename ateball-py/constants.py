from types import SimpleNamespace
from pathlib import Path
import json

with open(str(Path("ateball-py", "game_constants.json")), encoding='utf-8') as json_data:
    constants = json.load(json_data, object_hook=lambda d: SimpleNamespace(**d))