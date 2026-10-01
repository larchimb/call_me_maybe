
import argparse
import os
from .parser import Parser
from .llm import Llm
import json
from typing import Any


def parse_arguments() -> argparse.Namespace:
    """Read the command line options.

    Returns:
        The options, with the default paths for the missing ones.
    """
    arg_parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls"
    )
    arg_parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="JSON file with the available functions",
    )
    arg_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="JSON file with the prompts to process",
    )
    arg_parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="JSON file where the results are written",
    )
    return arg_parser.parse_args()


def main() -> None:
    """Translate every prompt of the input file into a function call."""
    args = parse_arguments()
    parser = Parser(args.input, args.functions_definition)
    llm = Llm(parser.prompts, parser.functions)
    returned_json: list[dict[str, Any]] = []
    i = 0
    for item in parser.prompts:
        returned_json.append({})
        dic = returned_json[i]
        function = llm.find_the_function(
            item.prompt, parser.functions
            )
        dic["prompt"] = item.prompt
        dic["name"] = function.name
        dic["parameters"] = llm.find_parameters(item.prompt, function)
        i += 1
    try:
        directory = os.path.dirname(args.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(returned_json, f, indent=4)
    except OSError as e:
        raise Exception(f"[ERROR]: cannot write {args.output}: {e}")
