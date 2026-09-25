from llm_sdk import Small_LLM_Model
from models import FunctionsDefinitions
import numpy as np
# import json


class Llm:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()
        self.vocab: dict[str, int] = {}
        self.load_vocab()
        self.end_token = 151645

    def load_vocab(self) -> None:
        '''Load the vocab one time to avoid multiple encode'''
        vocab_size = len(self.llm.get_logits_from_input_ids([0]))
        for token_id in range(vocab_size):
            text = self.llm.decode([token_id])
            if text != "":
                self.vocab[text] = token_id

    def get_max_token(self, functions: list[FunctionsDefinitions]) -> None:
        '''To define the max token'''
        self.max_token = max([len(f.name) for f in functions])

    def find_the_function(self,
                          prompt: str,
                          functions: list[FunctionsDefinitions]
                          ) -> FunctionsDefinitions:
        '''Finding the function corresponding to the description'''
        self.get_max_token(functions)
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
        answer = self.answer_building(sentence, [f.name for f in functions])
        print(answer)
        for function in functions:
            if function.name == answer:
                return function
        raise Exception(f"[ERROR]: unknown function {answer}")

    def find_valid_tokens(self, answer: str, names: list[str]) -> list[int]:
        '''Recuperate tokens that keep `answer` '''
        '''a prefix of at least one function's name'''
        authorized: list[int] = []
        for name in names:
            if not name.startswith(answer):
                continue
            rest = name[len(answer):]
            if not rest:
                authorized.append(self.end_token)
            for i in range(1, len(rest) + 1):
                if rest[:i] in self.vocab:
                    authorized.append(self.vocab[rest[:i]])
        return authorized

    def answer_building(self, sentence: list[int], names: list[str]) -> str:
        '''The loop for the llm to create the answer'''
        answer = ""
        token = 0
        while token < self.max_token:
            authorized = self.find_valid_tokens(answer, names)
            if not authorized:
                raise Exception("[ERROR]: No tokens available to follow")
            next_token = self.generate_next_token(sentence, authorized)
            if next_token == self.end_token:
                return answer
            sentence.append(next_token)
            answer += self.llm.decode([next_token])
            token += 1
        raise Exception(f"[ERROR]: {self.max_token} tokens raised")

    def generate_next_token(self, sentence: list[int], authorized: list[int]) -> int:
        '''Return the next choosen token'''
        logits = np.array(self.llm.get_logits_from_input_ids(sentence))
        mask = np.full(len(logits), -np.inf)
        mask[authorized] = 0
        return int(np.argmax(logits + mask))

    def add_text_to_answer(self, sentence: list[int], text: str) -> None:
        '''Add text to the sentence without asking the model'''
        sentence.extend(self.llm.encode(text)[0].tolist())

    def find_parameters(self, prompt: str, function: FunctionsDefinitions) -> dict:
        '''Generate each parameter value with constrained decoding'''
        example: dict = {}
        for name, kind in function.parameters.items():
            example[f"{name}"] = kind.type
        prompt_text = (
             "<|im_start|>system\n"
             "Extract parameters valus from a prompt for the given function\n"
             f"function: {function.description}\n"
            #   "Don't apply the function, only takes parameters\n"
             f"Answer must following this parsing: {example}\n"
             "<|im_end|>\n"
             "<|im_start|>user\n"
             f"Prompt: \"{prompt}\"\n"
             "<|im_end|>\n"
             "<|im_start|>assistant\n"
             "<think>\n\n</think>\n\n"
            )
        sentence = self.llm.encode(prompt_text)[0].tolist()
        result: dict = {}
        separator = "{"
        for name, kind in function.parameters.items():
            key = f'{separator}"{name}": '
            separator = ", "
            if kind.type == "integer":
                self.add_text_to_answer(sentence, key)
                result[name] = int(self.generate_number(sentence, True))
            elif kind.type == "boolean":
                self.add_text_to_answer(sentence, key)
                answer = self.answer_building(sentence, ["true", "false"])
                if answer == "true":
                    result[name] = True
                else:
                    result[name] = False
            elif kind.type == "string":
                self.add_text_to_answer(sentence, key + '"')
                # result[name] = self.generate_string(sentence)
            else:
                self.add_text_to_answer(sentence, key)
                result[name] = float(self.generate_number(sentence, False))
            print("test")
        print(result)
        return result

    def generate_number(self, sentence: list[int], integer: bool) -> str:
        '''Generate a number token by token'''
        digits = [self.vocab[c] for c in "0123456789"]
        allowed: list[int] = []
        answer = ""
        if integer:
            has_a_point = True
        else:
            has_a_point = False
        for i in range(self.max_token):
            if i == 0:
                allowed = digits + [self.vocab["-"]]
            elif not has_a_point:
                allowed = digits + [self.vocab["."]]
            else:
                allowed = digits[:]
            if sentence[-1] in digits:
                allowed.extend([self.vocab[","], self.vocab["}"]])
            next_token = self.generate_next_token(sentence, allowed)
            if next_token == self.vocab["."]:
                has_a_point = True
            if next_token in [self.vocab[","], self.vocab["}"]]:
                return answer
            sentence.append(next_token)
            answer += self.llm.decode([next_token])
        raise Exception("[ERROR]: number too long")


