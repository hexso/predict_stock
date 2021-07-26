import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

class LSTMStock(nn.Module):

    def __init__(self):
        super(LSTMStock, self).__init__()

        self.fileName = None
        self.data = DataFrame()
        self.inputX = DataFrame()
        self.outY = DataFrame()
        self.trainRate = 0.9
        self.date = None
        self.x_std = None
        self.y_mm = None
        self.inputSize = 5
        self.outputSize = 1
        self.layerNum = 1
        self.hiddenDim = 128
        self.epochCnt = 100
        self.windowSize = 20
        self.lstm = nn.LSTM(self.inputSize, self.hiddenDim, self.layerNum, batch_first=True)
        self.hiddenLayer = nn.Linear(self.hiddenDim, self.outputSize)


    def setFile(self, filename, date):
        self.fileName = filename
        self.data = pd.read_csv(filename)
        self.date = date
        #날짜 정렬 등 데이터 처리
        self.dataProcessing()

    def setInput(self, inputlist):
        for col in inputlist:
            self.inputX[col] = self.data.loc[:,col]
        self.inputSize = len(inputlist)

    def setOutput(self, outlist):
        for col in outlist:
            self.outY[col] = self.data.loc[:,col]
        self.outputSize = len(outlist)

    def dataProcessing(self):
        self.data.sort_values(by=self.date).reset_index()
        for col in self.data.columns:
            if col != self.date and self.data[col].dtype == 'O':
                self.data[col] = self.data[col].str.replace(',','')
                self.data[col] = self.data[col].str.replace('M','000000')
                self.data[col] = self.data[col].str.replace('K','000')
                self.data[col] = self.data[col].str.replace('.','')
                self.data[col] = self.data[col].astype(float)

        self.data.index = self.data[self.date]
        self.data = self.data.drop(self.date, axis=1)

    def sliceWindow(self, stock_data):
        data_raw = stock_data
        data = []

        for index in range(len(data_raw)-self.windowSize):
            data.append(data_raw[index: index+self.windowSize])

        data = np.array(data)
        return data

    def forward(self, x):
        h0 = torch.zeros(self.layerNum, x.size(0), self.hiddenDim).requires_grad_()

        c0 = torch.zeros(self.layerNum, x.size(0), self.hiddenDim).requires_grad_()

        out, (hn,cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.hiddenLayer(out[:, -1, :])
        return out

    def run(self, model):
        mm_scaler = MinMaxScaler(feature_range=(-1, 1))
        std_scaler = StandardScaler()

        x_std = std_scaler.fit_transform(self.inputX)
        y_mm = mm_scaler.fit_transform(self.outY.values.reshape(-1,1))

        x_slice = self.sliceWindow(x_std)
        y_slice = y_mm[:-self.windowSize]
        print(x_slice.shape)
        print(y_slice.shape)
        total_size = len(x_slice)
        x_train = x_slice[:int(total_size*self.trainRate)]
        x_test = x_slice[int(total_size*self.trainRate):]

        y_train = y_slice[:int(total_size*self.trainRate)]
        y_test = y_slice[int(total_size*self.trainRate):]

        print('training size is {}'.format(x_train.shape))
        print('test size is {}'.format(x_test.shape))

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

        loss_function = nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(self.epochCnt)
        print(model)

        start_time = time.time()
        for t in range(self.epochCnt):
            y_train_pred = model(x_train)
            print(y_train_pred.shape)
            print(y_train_lstm.shape)
            loss = loss_function(y_train_pred, y_train_lstm)
            print('Epoch ', t, 'MSE: ', loss.item())
            hist[t] = loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        train_time = time.time() - start_time
        print('Training Time : {}'.format(train_time))
        plt.plot(hist, label='Training loss')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    #날짜를 정해서 불러오는 방법
    # start = (2020,1,1)
    # start = datetime.datetime(*start)
    # end = datetime.date.today()
    # df = pdr.DataReader('005930.KS', 'yahoo', start, end)

    lstm = LSTMStock()
    lstm.setFile('stocks/samsung.csv','Date')
    inputList = ['Open','High','Low','Close','Volume']
    outputList = ['Close']

    lstm.setInput(inputList)
    lstm.setOutput(outputList)
    lstm.run(lstm)