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

    def longer_name(self, functions: list[FunctionsDefinitions]) -> None:
        '''To define the max token'''
        self.max_token = max([len(f.name) for f in functions])

    def find_the_function(self,
                          prompt: str,
                          functions: list[FunctionsDefinitions]
                          ) -> FunctionsDefinitions:
        '''Finding the function corresponding to the description'''
        self.longer_name(functions)
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
            logits = self.llm.get_logits_from_input_ids(sentence)
            logits = np.array(logits)
            mask = np.full(len(logits), -np.inf)
            mask[authorized] = 0
            next_token = int(np.argmax(logits + mask))
            if next_token == self.end_token:
                return answer
            sentence.append(next_token)
            answer += self.llm.decode([next_token])
            token += 1
        raise Exception(f"[ERROR]: {self.max_token} tokens raised")



