from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Literal, Annotated

NonEmpty = Annotated[
	str,
	StringConstraints(strip_whitespace=True, min_length=1)
	]


class FunctionsPrompt(BaseModel):
	model_config = ConfigDict(extra="forbid")
	prompt: NonEmpty
	function_used: str = ""


class ParamSpec(BaseModel):
	type: Literal[
     "number",
     "integer",
     "string",
     "boolean"
     ]


class FunctionsDefinitions(BaseModel):
	name: NonEmpty
	description: NonEmpty
	parameters: dict[NonEmpty, ParamSpec]
	returns: ParamSpec
