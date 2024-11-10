from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset
from tqdm import tqdm
from Preprocessing import split_string, embed_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pickle
class IMDBDataset(Dataset):
    def __init__(self, dataset, device):
        self.labels = torch.tensor(dataset["label"], device=device)
        self.lengths = []
        self.data = []
        for text in tqdm(dataset["text"]):
          tokens = split_string(text)
          embeddings = embed_data(tokens)
          self.data.append(embeddings.to(device))
          self.lengths.append(len(embeddings))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]
        label = self.labels[idx]
        length = self.lengths[idx]

        return embedding, length, label

def create_dataloader(dataset, device, batch_size, shuffle=True):
    imdb_dataset = IMDBDataset(dataset, device)
    dataloader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    dataset = load_dataset("stanfordnlp/imdb").shuffle(42)
    train_dataset = dataset['train'][:3000]
    test_dataset = dataset['test'][:1000]
    train_dataloader = create_dataloader(train_dataset, device, 1)
    test_dataloader = create_dataloader(test_dataset, device, 1)
    # Collect all batches from DataLoader
    train_data_list = [batch for batch in train_dataloader]
    test_data_list = [batch for batch in test_dataloader]

    # Save using pickle
    with open('../data/train_dataloader_data.pkl', 'wb') as f:
        pickle.dump(train_data_list, f)
    print("DataLoader saved to train_dataloader_data.pkl")

    with open('../data/test_dataloader_data.pkl', 'wb') as f:
        pickle.dump(test_data_list, f)
    print("DataLoader saved to test_dataloader_data.pkl")





