from llm_sdk import Small_LLM_Model
import torch




llm = Small_LLM_Model()
text = "{"
test = llm.encode(text)
print(test)
text = "\""
test = llm.encode(text)
print(test)
# print(llm.encode("{\"name\": \"john\"}"))
# liste = [ 4913,   606,   788,   330, 47817,  9207]
# for i in liste:
#     print(llm.decode([i]))