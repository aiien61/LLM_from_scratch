"""Python3.10
"""
from typing import List
import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text: str = file.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text: List[int] = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size: int = 4
x = enc_sample[:context_size]
y = enc_sample[1:(context_size+1)]
print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size+1):
    context: List[int] = enc_sample[:i]
    desired: int = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))