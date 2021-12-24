import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
'''
Linear를 이용한 주식 예측 모델
Binary loss function을 통해 3%상승, 5%상승과 같은 특정 수치를 맞혔는지에 대한 모델
따라서 output값을 0 또는 1로 되어있는 두개의 결과값으로 나타나 있는 값을 줘야 한다.
'''


class Linear(nn.Module):

    def __init__(self, inputs=[], output='Change5', processor='cpu'):
        super(Linear, self).__init__()

        self.train_rate = 0.8
        self.input_size = len(inputs)
        self.output_size = 1
        self.epochCnt = 50
        self.batch_size = 64
        self.learning_rate = 0.001
        self.device = torch.device(processor)
        self.std_scaler = StandardScaler()

        self.inputs = inputs
        self.output = output

        # Number of input features is 12.
        self.layer_1 = nn.Linear(self.input_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam

    def dataProcessing(self, data):
        x = pd.DataFrame()
        for col in self.inputs:
            x = pd.concat([x,data[col]],axis=1)
        y = data[self.output]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=69)
        x_train = self.std_scaler.fit_transform(x_train)
        x_test = self.std_scaler.transform(x_test)
        return x_train, x_test, y_train, y_test

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

    def learn(self, model, x_train, y_train):
        x_train = torch.from_numpy(np.array(x_train)).type(torch.Tensor)
        y_train = torch.from_numpy(np.array(y_train)).type(torch.Tensor)
        train_data = trainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for e in range(1, self.epochCnt + 1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()

                y_pred = model(X_batch)

                loss = self.loss_function(y_pred, y_batch.unsqueeze(1))
                acc = self.binary_acc(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            if e%10 == 0:
                print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')


    def predict(self, model, x_test, y_test):
        test_data = testData(torch.FloatTensor(x_test))
        test_loader = DataLoader(dataset=test_data, batch_size=1)
        y_pred_list = []
        model.eval()
        with torch.no_grad():
            for x_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_test_pred = model(x_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        print(confusion_matrix(y_test, y_pred_list))
        print(classification_report(y_test, y_pred_list))
        return y_pred_list

    def save(self, model, filename='model.pt'):
        torch.save(model, filename)

    def load(self, filename=''):
        model = torch.load(filename)
        return model

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

if __name__ == '__main__':
    from utils.UtilStock import StockCal
    import pandas as pd

    stock_code = {}
    start = (2020,6,1)
    start = datetime.datetime(*start)
    end = datetime.date.today()

    #stock_code = {'samsung': '005930'}

    #주식리스트를 텍스트파일에서 불러온다.
    with open('../stocks.txt', 'r', encoding='cp949') as f:
        datas = f.readlines()
        for data in datas:
            data = data.strip('\n').split(':')
            stock_code[data[0]] = data[1]

    #주식데이터를 다운받는다.
    # for name, code in stock_code.items():
    #     data = fdr.DataReader(code,start=start, end=end)
    #     data.to_csv('stocks/' + name+'.csv')
    #
    # #주식데이터로 보조지표를 만들어 낸다.
    stockCal = StockCal()
    for stock_name in stock_code.keys():
        print(stock_name)
        df = pd.read_csv('stocks/'+stock_name +'.csv')
        df = stockCal.getStockInput(df)
        df.to_csv('stocks/'+stock_name +'.csv', index=None)

    inputs = ['Volume', 'MACD', 'OBV', 'SMA20', 'SMA5', 'STOCHK', 'STOCHD', 'BUPPER', 'BMIDDLE', 'BLOWER', 'up']
    model = Linear(inputs=inputs, output='Change3_tmw')
    for stock_name in stock_code.keys():
        data = pd.read_csv('stocks/'+ stock_name +'.csv')

        x_train, x_test, y_train, y_test = model.dataProcessing(data)
        model.learn(model, x_train, y_train)

    # model = model.load('model.pt')
    data = pd.read_csv('../stocks/신일전자.csv')
    data = stockCal.getStockInput(data)
    x_train, x_test, y_train, y_test = model.dataProcessing(data)
    y_pred_list = model.predict(model, x_train, y_train)

    # figure, axes = plt.subplots(figsize=(15, 6))
    # axes.xaxis_date()
    # data.index = data['Date']
    #
    # axes.plot(data[len(data) - len(y_test):].index, y_test, color='red', label='Real price')
    # axes.plot(data[len(data) - len(y_test):].index, y_pred_list, color='blue', label='Predict price')
    #
    # plt.xlabel('Date')
    # plt.ylabel('price')
    # plt.legend()
    # plt.show()
    # model.save(model)
    # model = lstm.load('model.pt')
    # lstm.predict(lstm,data)