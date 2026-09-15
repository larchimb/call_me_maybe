from llm_sdk import Small_LLM_Model
from models import FunctionsDefinitions


class Llm:
    def __init__(self):
        self.llm = Small_LLM_Model()

    def find_the_function(self,	prompt: str, functions: list[FunctionsDefinitions]) -> FunctionsDefinitions:
        '''Finding the function corresponding to the description'''
        prompt_text = (
         "<|im_start|>system\n"
         "Return a function name corresponding to the user's prompt\n"
         "Answer will parse as following: \"function's name\"\n"
         f"Functions: {functions}\n"
         "<|im_end|>\n"
         "<|im_start|>user\n"
         f"Description: \"{prompt}\"\n"
         "<|im_end|>\n"
         "<|im_start|>assistant\n"
         "<think>\n\n</think>\n\n"
        )
        sentence = self.llm.encode(prompt_text)[0].tolist()
        authorized_tokens = self.find_authorized_tokens(functions)
        answer = self.llm.decode(self.answer_construction(sentence, authorized_tokens))
        print(answer)
        for function in functions:
            if function.name == answer:
                function_choosed = function
        return function_choosed

    def find_parameters(
     self,
     prompt: str,
     function: FunctionsDefinitions,
     ) -> str:
        '''Finding the parameters corresponding to the function in the prompt'''
        example: dict = {}
        for name, kind in function.parameters.items():
            example[f"{name}"] = kind.type

        prompt_text = (
             "<|im_start|>system\n"
             "Extract parameters valus from a prompt for the given function\n"
             f"function: {function.description}\n"
             f"You must return exactly {len(function.parameters)} parameter(s)"
              "Don't apply the function, only takes parameters\n"
             f"Answer must following this parsing: {example}"
             "<|im_end|>\n"
             "<|im_start|>user\n"
             f"Prompt: \"{prompt}\"\n"
             "<|im_end|>\n"
             "<|im_start|>assistant\n"
             "<think>\n\n</think>\n\n"
            )
        sentence = self.llm.encode(prompt_text)[0].tolist()
        answer = self.llm.decode(self.answer_construction(sentence, [])).strip().strip('"')
        # print(answer)
        return answer

    def answer_construction(self, sentence: list[int], restreigned_code: list[int]) -> list[int]:
        '''The loop for the llm to create the answer'''
        t_answer = []
        token_last_word = 0
        while token_last_word != 151645:
            logits = self.llm.get_logits_from_input_ids(sentence)
            m = max(logits)
            sentence.append(logits.index(m))
            t_answer.append(logits.index(m))
            token_last_word = sentence[-1]
        return (t_answer)

    def find_authorized_tokens(self, functions: list[FunctionsDefinitions]) -> list[int]:
        '''Recuperate authorized tokens for function name'''
        authorized: list[int] = []
        for function in functions:
            tokens = self.llm.encode(function.name)[0].tolist()
            authorized.extend(tokens)
        return authorized