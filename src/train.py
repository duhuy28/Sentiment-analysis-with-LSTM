import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from src.model.LSTM import LSTM
from src.model.RNN import RNN
import pickle
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, num_epochs, loss_function, optimizer,
              train_dataloader, test_dataloader, device):
  epoch_loss_logger = []
  epoch_f1_scores =[]

  for epoch in range(num_epochs):
      print(f"\n Epoch {epoch+1} of {num_epochs}")
      # training
      model.train()
      train_loss = []
      print("\t Training progress: \n")
      for embedding, _, label in tqdm(train_dataloader):
            embedding = embedding.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(embedding[0])

            # calculate the loss and perform backprop
            loss = loss_function(output[-1].squeeze(dim=0), torch.tensor(float(label[0]), device=device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
      epoch_loss_logger.append(torch.mean(torch.tensor(train_loss)))


      # evaluation
      print("\n\t Evaluation progress: \n")
      model.eval()
      predictions = []
      targets = []
      with torch.no_grad():
            for embedding, _, label in tqdm(test_dataloader):
                test_prediction = round(model(embedding[0])[-1].item())
                predictions.append(float(test_prediction))
                targets.append(int(label[0]))
            train_f1_score = f1_score(predictions, targets)
            print("\t Test F1 Score in epoch " + str(epoch) + ": " + str(train_f1_score)
                  + " Train loss: " + str(epoch_loss_logger[epoch].item()))
            epoch_f1_scores.append(train_f1_score)

  return epoch_loss_logger, epoch_f1_scores


def visualize_data(data, title):
  # [ToDo]
    plt.plot(data[0],label='train_loss')
    plt.plot(data[1],label= 'f1_score')
    plt.title(title)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    ## load the data
    with open('../data/train_dataloader_data.pkl', 'rb') as f:
        train_dataloader = pickle.load(f)
    with open('../data/test_dataloader_data.pkl', 'rb') as f:
        test_dataloader = pickle.load(f)
    model = LSTM(50, 100, 1)
    model.to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
    num_epochs = 15
    epoch_loss_logger, epoch_f1_scores=train(model, num_epochs, loss_function, optimizer, train_dataloader, test_dataloader, device)
    visualize_data([epoch_loss_logger, epoch_f1_scores], 'LSTM')
    torch.save(model, 'lstm.pt')

