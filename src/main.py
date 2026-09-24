from parser import Parser
from pydantic import ValidationError
from llm import Llm
from spinner import Spinner


def main() -> None:
    parser = Parser()
    llm = Llm()
    returned_json: list[dict] = []
    i = 0
    for item in parser.prompts:
        returned_json.append({})
        dic = returned_json[i]
        # with Spinner(f"Searching the function {i + 1}"):
        function = llm.find_the_function(
            item.prompt, parser.functions
            )
        dic["prompt"] = item.prompt
        dic["name"] = function.name
        # with Spinner("Parameters extraction"):
        # dic["parameters"] = llm.find_parameters(item.prompt, function)
        i += 1
    # print(returned_json)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, Exception, KeyboardInterrupt) as e:
        print(e)
