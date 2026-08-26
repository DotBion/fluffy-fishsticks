import yfinance as yf
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def calculate_accuracy(predictions, targets, threshold):
    correct_predictions = np.abs(predictions - targets) <= threshold
    accuracy = np.mean(correct_predictions)
    return accuracy

# Evaluate the model
def evaluate_model(model, data_loader, threshold=0.05):
    model.eval()
    accuracies = []
    for batch_X, batch_y in data_loader:
        output = model(batch_X)
        predictions = output.detach().numpy()
        targets = batch_y.detach().numpy()
        accuracy = calculate_accuracy(predictions, targets, threshold)
        accuracies.append(accuracy)
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy

# Prepare data for RNN
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length+1]  # Adjusted to include the next price as well
        sequences.append(sequence)
    return np.array(sequences)

def get_acc_pred(tickerSymbol):
    try:
        tickerData = yf.Ticker(tickerSymbol)

        tickerDf = tickerData.history(period='1y')  # Set period to 1 year

        tickerDf.to_csv(tickerSymbol + '_Stock_Data.csv')
        df = pd.read_csv(tickerSymbol + '_Stock_Data.csv')
        selected_columns = ['Open','High','Low','Close']
        df_new = df[selected_columns]
        data = df[selected_columns].to_numpy()
    except:
        
        print("Error")
        return         

    with open('./model.pkl', 'rb') as f:
        model = pickle.load(f)

    #Input (size = (1, sequence lenght, 4))
    # random_array = torch.tensor(np.random.rand(1,20, 4),dtype = torch.float32)
    # print(random_array.shape)

    prediction = model(torch.tensor((data.reshape(1,data.shape[0],4)),dtype = torch.float32)).detach().numpy()

    SEQ_LENGTH = 3  # Adjust this according to your sequence length preference
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = data
    prices_scaled = scaler.fit_transform(prices)
    BATCH_SIZE=64

    X = create_sequences(prices_scaled, SEQ_LENGTH)
    y = X[:, -1]  # Get the last element of each sequence as y
    X = X[:, :-1]  # Remove the last element from each sequence to form X

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    accuracy = evaluate_model(model, test_loader, threshold=0.03)
    prediction = scaler.inverse_transform(prediction)[0]
    return prediction, accuracy


# In[10]:

# Define the ticker symbol
tickerSymbol = 'NVDA'
tickerSymbol = 'MSFT'
prediction, accuracy = get_acc_pred(tickerSymbol)
print("Accuracy: {}".format(accuracy))
print('Prediction: {}'.format(prediction))