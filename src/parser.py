import json
from pydantic import BaseModel, ValidationError
from models import FunctionsPrompt, FunctionsDefinitions


class Parser():
	def __init__(self) -> None:
		'''To initialize and parse informations'''
		self.prompts: list[FunctionsPrompt] = self.load_validated(
			"data/input/function_calling_tests.json", FunctionsPrompt
			)
		self.functions : list[FunctionsDefinitions] = self.load_validated(
			"data/input/functions_definition.json", FunctionsDefinitions
			)

	@staticmethod
	def load_validated(path: str, model: type[BaseModel]) -> list:
		'''Check json in entry and return parsed datas'''
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
		except FileNotFoundError as e:
			raise (e)
		except OSError as e:
			raise Exception(f"[ERROR]: {path} is locked")
		validated = []
		for i, item in enumerate(data):
			try:
				validated.append(model.model_validate(item))
			except ValidationError as e:
				raise Exception(f"[ERROR]: function {i + 1} \n{e}")
		return validated
