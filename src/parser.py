import json
from json import JSONDecodeError
from pydantic import BaseModel, ValidationError
from .models import FunctionsPrompt, FunctionsDefinitions
from typing import Any


class Parser():
    """Load and validate the input files.

    Attributes:
        prompts: The validated prompts to process.
        functions: The validated available functions.
    """

    def __init__(self, input_path: str, functions: str) -> None:
        """Load the prompts and the functions definitions.

        Args:
            input_path: Path of the JSON file containing the prompts.
            functions: Path of the JSON file containing the functions.

        Raises:
            Exception: If a file is invalid or contains no function.
        """
        self.prompts: list[FunctionsPrompt] = self.load_validated(
            input_path, FunctionsPrompt
            )
        self.functions: list[FunctionsDefinitions] = self.load_validated(
            functions, FunctionsDefinitions
            )
        if not self.functions:
            raise Exception("[ERROR]: Your file doesn't contain any function.")

    @staticmethod
    def load_validated(path: str, model: type[BaseModel]) -> list[Any]:
        """Read a JSON list and validate each element with a model.

        Args:
            path: Path of the JSON file to read.
            model: The pydantic model used to validate each element.

        Returns:
            The validated elements.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If the file cannot be read, is not valid JSON, is
                not a list, or contains an invalid element.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise Exception("[ERROR]: Json file isn't a list")
        except FileNotFoundError as e:
            raise (e)
        except JSONDecodeError:
            raise Exception("[ERROR]: Your file contain invalid json.")
        except OSError:
            raise Exception(f"[ERROR]: {path} is locked")
        validated = []
        for i, item in enumerate(data):
            try:
                validated.append(model.model_validate(item))
            except ValidationError as e:
                raise Exception(f"[ERROR]: function {i + 1} \n{e}")
        return validated
