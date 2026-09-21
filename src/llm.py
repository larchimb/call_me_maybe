from llm_sdk import Small_LLM_Model
from models import FunctionsDefinitions, ParamSpec
import numpy as np
import json


class Llm:
    def __init__(self):
        self.llm = Small_LLM_Model()

    def find_the_function(self,	prompt: str, functions: list[FunctionsDefinitions]) -> FunctionsDefinitions:
        '''Finding the function corresponding to the description'''
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
        sentence = self.llm.encode(prompt_text)[0].tolist()
        authorized_tokens = self.find_f_valid_tokens(functions)
        answer = self.llm.decode(self.answer_building(sentence, authorized_tokens))
        print(answer)
        for function in functions:
            if function.name == answer:
                function_choosed = function
        return function_choosed

    def answer_building(self, sentence: list[int], restreigned_code: list[int]) -> list[int]:
        '''The loop for the llm to create the answer'''
        start = len(sentence)
        token_last_word = 0
        token = 0
        while token_last_word != 151645 and token < 150:
            logits = self.llm.get_logits_from_input_ids(sentence)
            if not len(restreigned_code) == 1:
                logits = np.array(logits)
                mask = np.ones(len(logits), dtype=bool)
                mask[restreigned_code] = False
                logits[mask] = -np.inf
            next_token = int(np.argmax(logits))
            sentence.append(next_token)
            token_last_word = next_token
            token += 1
        return (sentence[start:])

    def find_f_valid_tokens(self, functions: list[FunctionsDefinitions]) -> list[int]:
        '''Recuperate authorized tokens for function name'''
        authorized: list[int] = []
        for function in functions:
            tokens = self.llm.encode(function.name)[0].tolist()
            authorized.extend(tokens)
        authorized.append(151645)
        return authorized

    def find_parameters(
         self,
         prompt: str,
         function: FunctionsDefinitions,
         ) -> dict:
        '''Finding the parameters corresponding to the function in the prompt'''
        dic_param = {}
        answer = None
        example: dict = {}
        for name, kind in function.parameters.items():
            example[name] = kind.type
        prompt_text = (
         "<|im_start|>system\n"
         "Extract parameters values from a prompt for the given function\n"
         f"function: {function.description}\n"
         f"You must return exactly {len(function.parameters)} parameter(s)"
          "Don't apply the function, only takes parameters\n"
         f"Answer must following this parsing: {json.dumps(example)}\n"
         "<|im_end|>\n"
         "<|im_start|>user\n"
         f"Prompt: \"{prompt}\"\n"
         "<|im_end|>\n"
         "<|im_start|>assistant\n"
         "<think>\n\n</think>\n\n"
        )
        sentence = self.llm.encode(prompt_text)[0].tolist()
        prompt_token = self.llm.encode(prompt)[0].tolist()
        kind = next(iter(function.parameters.values())).type
        authorized_tokens = self.find_tokens(prompt_token, kind)
        answer_token = self.p_str_answer_building(sentence, authorized_tokens)
        answer = self.llm.decode(answer_token)
        print(answer)
        dic_param = json.loads(answer)
        return dic_param

    def find_tokens(self, prompt: list[int], kind: str) -> list[int]:
        '''Recuperate authorized tokens for parameter'''
        authorized: list[int] = prompt
        if kind == "numbers":
            numb = [i for i in range(15, 25)]
            numb.extend([12, 13])
            # authorized = [i for i in authorized if i in numb]
        authorized.append(151645)
        return authorized

    def p_str_answer_building(self, sentence: list[int], restreigned_code: list[int]) -> list[int]:
        '''The loop for the llm to create the answer'''
        start = len(sentence)
        token_last_word = 0
        token = 0
        while token_last_word != 151645 and token < 150:
            logits = self.llm.get_logits_from_input_ids(sentence)
            # logits = np.array(logits)
            # mask = np.ones(len(logits), dtype=bool)
            # mask[restreigned_code] = False
            # logits[mask] = -np.inf
            next_token = int(np.argmax(logits))
            sentence.append(next_token)
            token_last_word = next_token
            token += 1
        return (sentence[start:])

    # authorized.extend([1866, 3846])
