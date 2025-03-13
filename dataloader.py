import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from rich import print

class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids: List[int] = []
        self.target_ids: List[int] = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt: str, batch_size: int, max_length: int, stride: int, shuffle: bool, drop_last: bool, num_workers: int):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


def test_drive(batch_size: int, max_length: int, stride: int):
    print(f"batch_size={batch_size}, max_length={max_length}, stride={stride}")
    config: Dict[str, int] = {
        "batch_size": batch_size,
        "max_length": max_length, 
        "stride": stride, 
        "shuffle": False, 
        "drop_last": True, 
        "num_workers": 0
    }
    dataloader = create_dataloader_v1(raw_text, **config)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)
    print("-"*10)

def main():
    batch_size: int = 8
    max_length: int = 4
    stride: int = 4

    print(f"batch_size={batch_size}, max_length={max_length}, stride={stride}")
    config: Dict[str, int] = {
        "batch_size": batch_size,
        "max_length": max_length,
        "stride": stride,
        "shuffle": False,
        "drop_last": True,
        "num_workers": 0
    }

    dataloader = create_dataloader_v1(raw_text, **config)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs: \n", inputs)
    print("\nTargets:\n", targets)


if __name__ == "__main__":
    test_drive(batch_size=1, max_length=4, stride=1)
    test_drive(batch_size=1, max_length=2, stride=2)
    test_drive(batch_size=1, max_length=8, stride=2)
    test_drive(batch_size=2, max_length=4, stride=1)
    main()
