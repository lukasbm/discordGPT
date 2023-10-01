import csv
import os.path
import torch
from torch.utils.data import Dataset
from typing import List
import re


def load_channels() -> List[str]:
    with open("channels.txt", "r") as f:
        res = f.readlines()
    res = map(lambda x: x.replace("\n", ""), res)
    res = map(lambda x: str("c" + x), res)
    return list(res)


def channel_text(channel: str) -> str:
    text = ""
    with open(os.path.join("package", "messages", channel, "messages.csv"), newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reversed(list(reader)):
            c, att = row[-2], row[-1]

            # broken content because of delimiter stuff or something
            if c == "Contents" or att == "Attachments":
                continue

            # remove custom emotes or pings
            c = re.sub(r"<.*>", "", c)

            # remove urls
            c = re.sub(r"https?://\S+", "", c, flags=re.MULTILINE | re.DOTALL)

            # remove empty lines
            c = re.sub(r"^\s*\n*", "", c, flags=re.MULTILINE | re.DOTALL)

            # remove empty / spam message
            if c == "" or len(c) < 5:
                continue

            # annotate if it was a caption
            if att != "":
                text += "[a] "

            text += c
            text += "\n\n"
    return text


def load_text(channels: List[str]) -> str:
    return "\n\n".join([channel_text(ch) for ch in channels])


class MessagesDataset(Dataset):
    def __init__(self, data):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self._data = data
        self._vocab_size = vocab_size
        self._bock_size = 128
        self._data_size = data_size

    def __len__(self):
        pass

    @property
    def block_size(self):
        return self._bock_size

    @property
    def data_size(self):
        return self._data_size

    @property
    def vocab_size(self):
        return self._vocab_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self._data[idx: idx + self.block_size + 1]
        print(chunk)
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    text = load_text(load_channels())
    ds = MessagesDataset(text)
    print(ds[0])
    print("=========")
    print(ds[1])
