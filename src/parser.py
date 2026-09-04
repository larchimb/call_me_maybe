import json


class Parser():
    def __init__(self) -> None:
        with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
            data_txt = json.load(f)
        print(data_txt)
        # data = json.loads(data_txt)
        # print(data)
        with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
            data_txt = json.load(f)
        print(f"\n{data_txt}")