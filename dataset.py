from torch.utils.data import Dataset
import pandas as pd


class MessagesDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("package/messages/c704284266288250941/messages.csv", sep=",")
        print(df["Contents"])

    def __len__(self):
        pass

    @property
    def data_size(self):
        return 5

    @property
    def vocab_size(self):
        return 5

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    MessagesDataset()
