import json
import numpy as np
from llm_sdk import Small_LLM_Model
from models import FunctionsDefinitions

class Llm:
    END_TOKEN = 151645  # <|im_end|>
    MAX_VALUE_TOKENS = 100

    def __init__(self) -> None:
        self.llm = Small_LLM_Model()
        self.vocab: dict[str, int] = {}
        self.load_vocab()
        self.number_tokens = {
            text: token_id for text, token_id in self.vocab.items()
            if all(c in "0123456789.-" for c in text)
        }
        self.string_ids = list(self.vocab.values())
        self.stop_ids = [self.vocab[","], self.vocab["}"]]

    def load_vocab(self) -> None:
        '''Load the vocab one time to avoid multiple encode'''
        vocab_size = len(self.llm.get_logits_from_input_ids([0]))
        for token_id in range(vocab_size):
            text = self.llm.decode([token_id])
            if text != "":
                self.vocab[text] = token_id

    def pick_token(self, sentence: list[int], allowed: list[int]) -> int:
        '''Return the best token among the allowed ones'''
        logits = np.array(self.llm.get_logits_from_input_ids(sentence))
        mask = np.full(len(logits), -np.inf)
        mask[allowed] = 0.0
        return int(np.argmax(logits + mask))

    def force(self, sentence: list[int], text: str) -> None:
        '''Add text to the sentence without asking the model'''
        sentence.extend(self.llm.encode(text)[0].tolist())

    # ------------------------------------------------------------------
    # Function name
    # ------------------------------------------------------------------

    def find_the_function(self, prompt: str,
                          functions: list[FunctionsDefinitions]
                          ) -> FunctionsDefinitions:
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
        answer = self.answer_building(sentence, [f.name for f in functions])
        for function in functions:
            if function.name == answer:
                return function
        raise Exception(f"[ERROR]: unknown function {answer}")

    def find_valid_tokens(self, answer: str, names: list[str]) -> list[int]:
        '''Tokens that keep `answer` a prefix of at least one name'''
        authorized: list[int] = []
        for name in names:
            if not name.startswith(answer):
                continue
            rest = name[len(answer):]
            if not rest:
                authorized.append(self.END_TOKEN)
            for i in range(1, len(rest) + 1):
                if rest[:i] in self.vocab:
                    authorized.append(self.vocab[rest[:i]])
        return authorized

    def answer_building(self, sentence: list[int], names: list[str]) -> str:
        '''Generate one of the names, token by token'''
        answer = ""
        max_tokens = max(len(name) for name in names) + 1
        for _ in range(max_tokens):
            authorized = self.find_valid_tokens(answer, names)
            if not authorized:
                raise Exception("[ERROR]: No tokens available to follow")
            next_token = self.pick_token(sentence, authorized)
            if next_token == self.END_TOKEN:
                return answer
            sentence.append(next_token)
            answer += self.llm.decode([next_token])
        raise Exception(f"[ERROR]: limit of {max_tokens} tokens reached")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def find_parameters(self, prompt: str, function: FunctionsDefinitions
                        ) -> dict[str, float | int | str | bool]:
        '''Generate each parameter value with constrained decoding'''
        params = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in function.parameters.items()
        )
        prompt_text = (
            "<|im_start|>system\n"
            "Extract the arguments of the function from the user's prompt "
            "and answer in JSON.\n"
            "Example:\n"
            "Function: fn_divide(dividend: number, divisor: number)\n"
            "Prompt: \"Divide 10 by 4\"\n"
            "Answer: {\"dividend\": 10, \"divisor\": 4}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Function: {function.name}({params})\n"
            f"Description: {function.description}\n"
            f"Prompt: \"{prompt}\"\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        sentence = self.llm.encode(prompt_text)[0].tolist()
        result: dict[str, float | int | str | bool] = {}
        separator = "{"
        for name, spec in function.parameters.items():
            key = f'{separator}"{name}": '
            separator = ", "
            if spec.type == "string":
                self.force(sentence, key + '"')
                result[name] = self.generate_string(sentence)
            elif spec.type == "boolean":
                self.force(sentence, key)
                answer = self.answer_building(sentence, ["true", "false"])
                result[name] = answer == "true"
            elif spec.type == "integer":
                self.force(sentence, key)
                result[name] = int(self.generate_number(sentence, True))
            else:
                self.force(sentence, key)
                result[name] = float(self.generate_number(sentence, False))
        return result

    @staticmethod
    def is_number_prefix(text: str, integer: bool) -> bool:
        '''Check if text can still become a valid number'''
        if "-" in text[1:]:
            return False
        if integer and "." in text:
            return False
        return text.count(".") <= 1

    @staticmethod
    def is_complete_number(text: str, integer: bool) -> bool:
        '''Check if text is already a valid number'''
        try:
            if integer:
                int(text)
            else:
                float(text)
            return True
        except ValueError:
            return False

    def generate_number(self, sentence: list[int], integer: bool) -> str:
        '''Generate a number token by token'''
        value = ""
        for _ in range(self.MAX_VALUE_TOKENS):
            allowed = [
                token_id for text, token_id in self.number_tokens.items()
                if self.is_number_prefix(value + text, integer)
            ]
            if self.is_complete_number(value, integer):
                allowed.extend(self.stop_ids)
            next_token = self.pick_token(sentence, allowed)
            if next_token in self.stop_ids:
                return value
            sentence.append(next_token)
            value += self.llm.decode([next_token])
        raise Exception("[ERROR]: number too long")

    def generate_string(self, sentence: list[int]) -> str:
        '''Generate a string until the closing quote'''
        raw = ""
        for _ in range(self.MAX_VALUE_TOKENS):
            next_token = self.pick_token(sentence, self.string_ids)
            text = self.llm.decode([next_token])
            if '"' in text:
                raw += text[:text.index('"')]
                self.force(sentence, '"')
                try:
                    return str(json.loads(f'"{raw}"'))
                except json.JSONDecodeError:
                    return raw
            sentence.append(next_token)
            raw += text
        raise Exception("[ERROR]: string too long")
