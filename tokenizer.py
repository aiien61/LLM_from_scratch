from typing import List, Dict
from icecream import ic
import re

### sample tokeniser
# text: str = "Hello, world. This, is a test."
# result: List[str] = re.split(r"(\s)", text)
# ic(result)

# result = re.split(r"([,.]|\s)", text)
# ic(result)

# result = [item for item in result if item.strip()]
# ic(result)

# text = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# ic(result)

# processing the whole story text
file_path: str = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    raw_text: str = file.read()

preprocessed: List[str] = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# ic(len(preprocessed))
# ic(preprocessed[:30])

### 將 token 轉換為 token ID
all_words: List[str] = sorted(set(preprocessed))
vocab_size: int = len(all_words)
ic(vocab_size)

# 建立詞彙表
vocab: Dict[str, int] = dict(zip(all_words, range(vocab_size)))
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        preprocessed: List[str] = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids: List[int] = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: List[int]) -> List[str]:
        text: str = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # remove spaces before punctuation
        return text

tokenizer: SimpleTokenizerV1 = SimpleTokenizerV1(vocab)
text: str = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids: List[int] = tokenizer.encode(text)
ic(ids)
ic(tokenizer.decode(ids))


try:
    text = "Hello, do you like tea?"
    ic(tokenizer.encode(text))
except KeyError as e:
    ic(e)