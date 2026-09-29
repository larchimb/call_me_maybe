from pydantic import BaseModel, ConfigDict
from typing import Literal


class FunctionsPrompt(BaseModel):
    """A prompt to translate into a function call.

    Attributes:
        prompt: The natural language request.
        function_used: Name of the function chosen for the prompt.
    """

    model_config = ConfigDict(
     extra="forbid",
     str_strip_whitespace=True,
     str_min_length=1
     )
    prompt: str
    function_used: str = ""


class ParamSpec(BaseModel):
    """The type of a parameter or of a return value.

    Attributes:
        type: One of "number", "integer", "string" or "boolean".
    """

    type: Literal[
     "number",
     "integer",
     "string",
     "boolean"
     ]


class FunctionsDefinitions(BaseModel):
    """A function the model can call.

    Attributes:
        name: The function name.
        description: What the function does.
        parameters: The parameter names mapped to their types.
        returns: The type of the returned value.
    """
    model_config = ConfigDict(
         extra="forbid",
         str_strip_whitespace=True,
         str_min_length=1
         )
    name: str
    description: str
    parameters: dict[str, ParamSpec]
    returns: ParamSpec
