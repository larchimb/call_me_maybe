*This project has been created as part of the 42 curriculum by larchimb.*

# call me maybe

## Description

`call me maybe` translates natural language requests into structured function
calls. Given a prompt such as *"What is the sum of 2 and 3?"* and a list of
available functions, the program does not answer `5`: it returns the function
to call and its typed arguments.

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

It relies on a small language model (Qwen3-0.6B) and on **constrained
decoding**: at each generation step, every token that would break the expected
format is forbidden, so the output is always valid JSON that follows the
schema of `functions_definition.json`.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install    # uv sync: installs numpy, pydantic, torch, mypy, flake8...
make run        # uv run python -m src
make debug      # runs the program under pdb
make lint       # flake8 + mypy with the flags required by the subject
make clean      # removes __pycache__ and .mypy_cache
```

The model is downloaded from Hugging Face on the first run.

### Example usage

```bash
# default paths: data/input/ -> data/output/function_calling_results.json
uv run python -m src

# custom paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

## Algorithm explanation

The program works in two steps for each prompt. In both, the model only
produces logits; the program decides which tokens are allowed and picks the
best one with `argmax` after setting every forbidden logit to `-inf`.

**Vocabulary.** At startup, every token id is decoded once to build a
`text -> id` dictionary. This lets the program find, in constant time, which
token has a given text.

**1. Choosing the function (prefix constraint).** The model generates the
function name token by token. A token is allowed only if the text generated so
far plus this token is still the beginning of at least one function name. The
end token `<|im_end|>` is allowed only when the text is exactly a complete
name. The generated name is therefore always an
existing function, and the choice is made by the LLM.

**2. Extracting the parameters (type constraint).** The JSON structure
(`{`, keys, `:`, `,`, quotes) is written by the program, not by the model.
The model only generates the values, one parameter at a time, with a
constraint depending on the type:

| Type | Allowed tokens | Stop |
|---|---|---|
| `number` / `integer` | a leading space or ` -`, then digits, and one `.` for numbers | `,` or `}`, only right after a digit |
| `boolean` | prefixes of `true` / `false` (same mechanism as the names) | end token |
| `string` | any token | first `"` |

Each generated value stays in the context, so the model sees the previous
arguments when it generates the next one. The values are then converted to
`float`, `int`, `bool` or `str`, and the output file is written with
`json.dump`, which guarantees valid JSON.

## Design decisions

- **The program writes the structure, the model writes the values.** Asking
  the model to produce the braces and keys would only add ways to fail.
- **Chat template with an empty `<think>` block.** Qwen3 is a chat model;
  the empty thinking block puts it directly in answer mode.
- **A single example in the parameter prompt.** It shows the model that a
  description (*"all spaces"*) must be translated into a regex (`\s`). It uses
  a made-up request so that nothing is hardcoded from the test set.
- **Leading space handled as a token.** In Qwen's tokenizer, `-5` after a
  space is tokenized as ` -` + `5`. Forcing the space separately made the
  model drop the minus sign, so the first step allows ` ` or ` -`.
- **pydantic models** validate the input files (`extra="forbid"`, non-empty
  strings, allowed types) before the model is loaded.
- **One token limit**, computed once from the longest prompt and the longest
  function name, protects every generation loop against infinite loops.

## Performance analysis

On the provided test set (11 prompts):

- **Function selection:** 11/11.
- **Argument extraction:** 18/19 parameters correct. The only error is
  *"with asterisks"* returned as `asterisk` instead of `*`.
- **JSON validity:** 100% by construction, since the structure is written by
  the program and every value is constrained to its type.
- **Speed:** the whole test set runs in less than 10s on CPU. Building the
  vocabulary takes a few seconds at startup.

## Challenges faced

- **Constraigned decoding.** A first version with only a prompt manage to reach
  the 90% rate, but it didn't use constraigned decoding.
- **Hybrid function names.** The next version allowed every token of every
  name in any order, which produced names like `fn_reverse_numbers`. Fixed
  with the prefix constraint.
- **Unclosed strings.** The closing quote was not written in the context,
  so the model saw broken JSON and the next values were wrong (`"dog, "`).
- **Negative numbers** were lost because of the tokenization of ` -`.
- **Regex parameters.** The model copied words from the prompt (`numbers`)
  instead of writing a pattern. The example in the prompt fixed it.

## Testing strategy

- Running the provided prompts and checking each function name, each value
  and each type by hand.
- Edge cases on the input files: missing file, invalid JSON, a JSON object
  instead of a list, empty function list, invalid types, empty strings.
- Edge cases on the prompts: negative numbers, quotes and backslashes in the
  values, prompts matching no function.
- `make lint` (flake8 and mypy) on every change.

## Resources

- [LLM Global explanation](https://mikexcohen.substack.com/p/llm-breakdown-16-tokenization-words)
- [Qwen3 model card](https://qwen.readthedocs.io/en/latest/)
- [JSON's explanations](https://www.datacamp.com/tutorial/json-data-python?utm_cid=23781701478&utm_aid=196565213035&utm_campaign=260417_1-ps-dscia~amx-tofu~python_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9056032-&utm_mtd=p-c&utm_kw=json%20data%20python&utm_source=google&utm_medium=paid_search&utm_content=ps-dscia~emea-en~amx~tofu~tutorial~python&gad_source=1&gad_campaignid=23781701478&gbraid=0AAAAADQ9WsEEBLP8M4Q1vp9LfWM4dK6MK&gclid=Cj0KCQjwh4TVBhCWARIsAG0czmqtg00zJwkwLvOgl3C39PuhNUQmn_UpUA4CBj_EWPBs--MXGeJJhIEaAoCeEALw_wcB2)
- [pydantic documentation](https://docs.pydantic.dev/)

### Use of AI

Claude (Anthropic) was used as an assistant during the project:

- **Explanations:** constrained decoding, tokenization (byte-level BPE),
  chat templates, Python and numpy details.
- **Code review:** finding bugs in the generation loops, mypy and flake8
  issues, the Makefile and the package imports.
- **Suggestions:** the prefix lookup for function names, the prompt example
  for regex parameters.
- **Writing:** the docstrings and a first draft of this README.

All the code was read, tested and adapted before being kept.
