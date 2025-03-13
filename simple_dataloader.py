"""Python3.10
"""
import os
import requests

# source: https://en.wikisource.org/wiki/The_Verdict
url: str = ("https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")

file_path: str = "the-verdict.txt"


if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    print("File already has content. Skipping fetch.")
else:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"Text saved to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Requests failed: {e}")

with open(file_path, "r", encoding="utf-8") as file:
    raw_text: str = file.read()

print(f"Total number of character: {len(raw_text)}")
print(raw_text[:99])