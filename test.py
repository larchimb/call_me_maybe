from llm_sdk import Small_LLM_Model


llm = Small_LLM_Model()
text = "."
test = llm.encode(text)
print(test)