from llm_sdk import Small_LLM_Model
from parser import Parser
from pydantic import ValidationError
from models import FunctionsDefinitions, Answer


def find_the_function(
    prompt: str, functions: list[FunctionsDefinitions]) -> str:
    llm = Small_LLM_Model()
    test = llm.encode(f"Find the closest function in the list {functions} accordind to her description from the next description \"{prompt}\"."
                      "Your answer must be one of the function name")
    sentence = test[0].tolist()
    answer = []
    token_last_word = 0
    i = 0
    while token_last_word != 13:
        out = llm.get_logits_from_input_ids(sentence)
        m = max(out)
        sentence.append(out.index(m))
        answer.append(out.index(m))
        token_last_word = sentence[-1]
    print(llm.decode(answer))

def main() -> None:
    parser = Parser()
    # for item in parser.prompts:
    find_the_function(parser.prompts[0].prompt, parser.functions)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, Exception, KeyboardInterrupt) as e:
        print(e)
