from llm_sdk import Small_LLM_Model
import numpy

txt = "Ben est grand"

llm = Small_LLM_Model()

test = llm.encode(txt)
print(test)
entry = test[0].tolist()
out = llm.get_logits_from_input_ids(entry)
m = max(out)
m2 = i in max(out) if i != m
print(llm.decode([out.index(m)]))
print(llm.decode(test))
for i in test[0]:
    print(llm.decode(i))
