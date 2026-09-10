from llm_sdk import Small_LLM_Model
from parser import Parser
from pydantic import ValidationError
from models import FunctionsDefinitions


def find_the_function(
    prompt: str,
    functions: list[FunctionsDefinitions],
    llm: Small_LLM_Model
    ) -> FunctionsDefinitions | None:
    prompt_text = (
     "<|im_start|>system\n"
     "Return a function name corresponding to the user's prompt\n"
     "Answer will parse as following : \"function's name\""
     f"Functions: {functions}\n"
     "<|im_end|>\n"
     "<|im_start|>user\n"
     f"Description: \"{prompt}\"\n"
     "<|im_end|>\n"
     "<|im_start|>assistant\n"
     "<think>\n\n</think>\n\n"
    )
    sentence = llm.encode(prompt_text)[0].tolist()
    t_answer = []
    token_last_word = 0
    while token_last_word != 151645:
        logits = llm.get_logits_from_input_ids(sentence)
        m = max(logits)
        sentence.append(logits.index(m))
        t_answer.append(logits.index(m))
        token_last_word = sentence[-1]
    answer = llm.decode(t_answer).strip().strip('"')
    for function in functions:
        if function.name == answer:
            return function
    return


def find_parameters(
    prompt: str,
    function: FunctionsDefinitions,
    llm: Small_LLM_Model) -> None:
    prompt_text = (
         "<|im_start|>system\n"
         f"Return parameters for {function} from user's prompt\n"
         "Answer will parse as following : {function.parameters}}"
         "<|im_end|>\n"
         "<|im_start|>user\n"
         f"Description: \"{prompt}\"\n"
         "<|im_end|>\n"
         "<|im_start|>assistant\n"
         "<think>\n\n</think>\n\n"
        )
    sentence = llm.encode(prompt_text)[0].tolist()
    t_answer = []
    token_last_word = 0
    while token_last_word != 151645:
        logits = llm.get_logits_from_input_ids(sentence)
        m = max(logits)
        sentence.append(logits.index(m))
        t_answer.append(logits.index(m))
        token_last_word = sentence[-1]


def main() -> None:
    parser = Parser()
    returned_json: list[dict] = []
    llm = Small_LLM_Model()
    i = 0
    for item in parser.prompts:
        returned_json.append({})
        dic = returned_json[i]
        function = find_the_function(
            item.prompt, parser.functions, llm
            )
        if not function:
            raise Exception("[ERROR] : No function found for \"{item.prompt}\"")
        dic["prompt"] = item.prompt
        dic["name"] = function.name
        dic["parameters"] = find_parameters(item.prompt, function, llm)
        i += 1
    # print(returned_json)


if __name__ == "__main__":
    # try:
    main()
    # except (ValidationError, Exception, KeyboardInterrupt) as e:
        # print(e)
