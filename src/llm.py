from llm_sdk import Small_LLM_Model
from models import FunctionsDefinitions
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
             f"Answer must following this parsing: {json.dumps(example)}\n"
             "<|im_end|>\n"
             "<|im_start|>user\n"
             f"Prompt: \"{prompt}\"\n"
             "<|im_end|>\n"
             "<|im_start|>assistant\n"
             "<think>\n\n</think>\n\n"
            )
        sentence = self.llm.encode(prompt_text)[0].tolist()
        authorized_tokens = self.find_p_valid_tokens(function, example)
        answer = self.llm.decode(self.p_answer_building(sentence, authorized_tokens))
        print(answer)
        return answer

    def find_p_valid_tokens(self, function: FunctionsDefinitions, example: dict) -> list[int]:
        '''Recuperate authorized tokens for function name'''
        authorized: set[int] = set()
        for param in function.parameters.values():
            authorized.update(self.llm.encode(json.dumps(list(example.keys())))[0].tolist())
            if param.type == "number":
                authorized.update([i for i in range(15, 25)])
                authorized.update([11, 12, 13])
                authorized.update([1, 25, 90, 92])
            elif param.type == "integer":
                authorized.update([i for i in range(15, 25)])
                authorized.add(12)
            elif param.type == "boolean":
                authorized.update([1866, 3846])
            elif param.type == "string":
                authorized = set()
                authorized.add(151645)
                break
        authorized.add(151645)
        return list(authorized)

    def p_answer_building(self, sentence: list[int], restreigned_code: list[int]) -> list[int]:
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

