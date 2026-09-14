from llm_sdk import Small_LLM_Model
from parser import Parser
from pydantic import ValidationError
from models import FunctionsDefinitions

def answer_construction(llm: Small_LLM_Model, sentence: list[int]) -> list[int]:
    '''The loop for the llm to create the answer'''
    t_answer = []
    token_last_word = 0
    while token_last_word != 151645:
        logits = llm.get_logits_from_input_ids(sentence)
        m = max(logits)
        sentence.append(logits.index(m))
        t_answer.append(logits.index(m))
        token_last_word = sentence[-1]
    return (t_answer)


def find_the_function(
    prompt: str,
    functions: list[FunctionsDefinitions],
    llm: Small_LLM_Model
    ) -> FunctionsDefinitions | None:
    '''Finding the function corresponding to the description'''
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
    answer = llm.decode(answer_construction(llm, sentence)).strip().strip('"')
    print(answer)
    for function in functions:
        if function.name == answer:
            return function
    return


def find_parameters(
    prompt: str,
    function: FunctionsDefinitions,
    llm: Small_LLM_Model) -> str:
    '''Finding the parameters corresponding to the function in the prompt'''
    example: dict = {}
    for name, kind in function.parameters.items():
        example[f"{name}"] = kind.type
        
    prompt_text = (
         "<|im_start|>system\n"
         "Extract parameters valus from a prompt for the given function\n"
         f"function: {function.description}\n"
         f"You must return exactly {len(function.parameters)} parameter(s)"
        #  "Answer ONLY with a single-line JSON object mapping each parameter "
        #  "to its value, with no explanation and no extra text.\n"
          "Don't apply the function, only takes parameters\n"
         f"Answer must following this parsing : {example}"
         "<|im_end|>\n"
         "<|im_start|>user\n"
         f"Prompt: \"{prompt}\"\n"
         "<|im_end|>\n"
         "<|im_start|>assistant\n"
         "<think>\n\n</think>\n\n"
        )
    sentence = llm.encode(prompt_text)[0].tolist()
    answer = llm.decode(answer_construction(llm, sentence)).strip().strip('"')
    print(answer)
    return answer


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
        # if i == 1:
        #     break
    print(returned_json)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, Exception, KeyboardInterrupt) as e:
        print(e)
