"""Python3.10
"""
from importlib.metadata import version
from typing import List
from toolz import pipe
import tiktoken
import unittest

def test_drive():
    print(f"tiktoken version: {version("tiktoken")}")

    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    text: str = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    integers: List[int] = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings: str = tokenizer.decode(integers)
    print(strings)


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    def test_bpe(self):
        text: str = "Akwirw ier"
        actual: str = pipe(text,
                           lambda x: self.tokenizer.encode(x, allowed_special={"<|endoftext|>"}),
                           lambda x: self.tokenizer.decode(x))
        expected: str = "Akwirw ier"
        self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
