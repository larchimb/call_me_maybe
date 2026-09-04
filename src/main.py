# from llm_sdk import Small_LLM_Model
# import numpy
from parser import Parser

txt = "Ali est grand."

# llm = Small_LLM_Model()
# i = 0
# test = llm.encode(txt)
# print(test)
# sentence = test[0].tolist()
# token_last_word = 0
# while token_last_word != 13:
#     out = llm.get_logits_from_input_ids(sentence)
#     m = max(out)
#     sentence.append(out.index(m))
#     token_last_word = sentence[-1]

# print(llm.decode(sentence))
def main() -> None:
    parser = Parser()

main()
