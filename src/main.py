from llm_sdk import Small_LLM_Model
from parser import Parser
from pydantic import ValidationError
from models import FunctionsDefinitions, Answer


def find_the_function(
    prompt: str, functions: list[FunctionsDefinitions]) -> None:
    llm = Small_LLM_Model()
    prompt_text = (
     "<|im_start|>system\n"
     "Return a function name corresponding to the user's prompt\n"
     f"Functions: {functions}\n"
     "<|im_end|>\n"
     "<|im_start|>user\n"
     f"Description: \"{prompt}\"\n"
     "<|im_end|>\n"
     "<|im_start|>assistant\n"
     "<think>\n\n</think>\n\n"
    )
    test = llm.encode(prompt_text)
    # test = llm.encode(f"Find the function in {functions} whose descrition corresponding to {prompt}."
    #                   "Your answer must start with \"\"fn\"")
    sentence = test[0].tolist()
    answer = []
    token_last_word = 0
    i = 0
    while token_last_word != 151645:
        out = llm.get_logits_from_input_ids(sentence)
        m = max(out)
        sentence.append(out.index(m))
        answer.append(out.index(m))
        token_last_word = sentence[-1]
    print(llm.decode(answer))

def main() -> None:
    parser = Parser()
    for item in parser.prompts:
        find_the_function(item.prompt, parser.functions)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, Exception, KeyboardInterrupt) as e:
        print(e)
