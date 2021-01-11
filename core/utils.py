import os
import json

from pathlib import Path


class get:
    def __init__(self):
        ROOT_DIR = self.root_dir()
        # print(ROOT_DIR)
        print(ROOT_DIR)
        meta_path = os.path.join(ROOT_DIR, '_metadata.json')
        with open(meta_path, 'r') as fp:
            data = json.load(fp)
            # print(data)
        for key, value in data.items():
            setattr(self, key, value)

    def root_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    def run_dir(self) -> Path:
        ROOT_DIR = get().root_dir()
        RUN_DIR = os.path.join(ROOT_DIR, self.runID)
        return RUN_DIR


# ROOT_DIR = get().root_dir()
# RUN_DIR = get().run_dir()
# thing = get().runID
# print(thing)
# print(RUN_DIR)
