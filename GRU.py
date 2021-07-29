import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time


class GRUStock(nn.Module):

    def __init__(self, output='Close',minmax=[], robust=[], origin=[], std=[],processor='cpu'):
        super(GRUStock, self).__init__()
        self.fileName = None
        self.trainRate = 0.9
        self.inputSize = len(minmax) + len(robust) + len(origin) + len(std)
        self.outputSize = 1
        self.layerNum = 1
        self.hiddenDim = 128
        self.epochCnt = 100
        self.windowSize = 20
        self.output = output
        self.device = torch.device(processor)
        # Scaler 리스트
        self.minmaxList = minmax
        self.stdList = std
        self.originList = origin
        self.robustList = robust
        self.scalerDataList = [self.minmaxList, self.stdList, self.robustList]
        self.minmaxScaler = MinMaxScaler()
        self.stdScaler = StandardScaler()
        self.robustScaler = RobustScaler()
        self.scalerList = [self.minmaxScaler, self.stdScaler, self.robustScaler]

        self.gru = nn.GRU(self.inputSize, self.hiddenDim, self.layerNum).to(self.device)
        self.hiddenLayer = nn.Linear(self.hiddenDim, self.outputSize).to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.layerNum, self.windowSize-1, self.hiddenDim).to(self.device).requires_grad_()
        out, hn = self.gru(x, h0.detach())

        out = self.hiddenLayer(out[:,-1])
        return out

    def sliceWindow(self, stock_data):
        data_raw = stock_data
        data = []

        for index in range(len(data_raw)-self.windowSize):
            data.append(data_raw[index: index+self.windowSize])

        return data

    def dataProcessing(self, data):
        new_data = pd.DataFrame()
        new_data['Date'] = data['Date']

        for idx, scalerData in enumerate(self.scalerDataList):
            for col in scalerData:
                reshape_data = data[col].values.reshape(-1,1)
                new_data[col] = self.scalerList[idx].fit_transform(reshape_data)

        for col in self.originList:
            new_data[col] = data[col]

        if 'Date' in data.columns:
            new_data = new_data.drop('Date',axis=1)
            new_data.index = data['Date']

        slice_data_x = self.sliceWindow(new_data)
        slice_data_y = self.sliceWindow(new_data[self.output])
        x_slice_data = np.array(slice_data_x)
        y_slice_data = np.array(slice_data_y)

        return x_slice_data, y_slice_data

    def learn(self, model, data):
        x_slice, y_slice = self.dataProcessing(data)

        total_size = len(x_slice)
        train_size = int(total_size * self.trainRate)

        x_train = x_slice[:train_size,:-1]
        x_test = x_slice[train_size:,:-1]

        y_train = y_slice[:train_size,-1]
        y_test = y_slice[train_size:,-1]

        x_train_torch = torch.from_numpy(x_train).type(torch.Tensor).to(self.device)
        x_test_torch = torch.from_numpy(x_test).type(torch.Tensor).to(self.device)
        y_train_torch = torch.from_numpy(y_train).type(torch.Tensor).to(self.device)
        y_test_torch = torch.from_numpy(y_test).type(torch.Tensor).to(self.device)

        loss_function = nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(self.epochCnt)
        print(model)

        start_time = time.time()
        for t in range(self.epochCnt):
            y_train_pred = model(x_train_torch)
            loss = loss_function(y_train_pred, y_train_torch)
            if t % 5 == 0:
                print('Epoch ', t, 'MSE: ', loss.item())
            hist[t] = loss.item()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        train_time = time.time() - start_time
        print('Training Time : {}'.format(train_time))


        y_test_pred = model(x_test_torch)

        #test결과 출력
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()
        axes.plot(data[len(data) - len(y_test):].index, y_test_pred.to('cpu').detach().numpy(), color='red', label='Real price')
        axes.plot(data[len(data) - len(y_test):].index, y_test_torch.to('cpu').detach().numpy(), color='blue', label='Predict price')

        plt.xlabel('Time')
        plt.ylabel('price')
        plt.legend()
        plt.show()