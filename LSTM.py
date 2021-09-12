import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import joblib
'''
LSTM을 이용한 주식 예측 모델
Binary loss function을 통해 3%상승, 5%상승과 같은 특정 수치를 맞혔는지에 대한 모델
따라서 output값을 0 또는 1로 되어있는 두개의 결과값으로 나타나 있는 값을 줘야 한다.
'''
class LSTMStock(nn.Module):

    def __init__(self, output='Close',minmax=[], robust=[], origin=[], std=[],processor='cpu'):
        super(LSTMStock, self).__init__()

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
        #Scaler 리스트
        self.minmaxList = minmax
        self.stdList = std
        self.originList = origin
        self.robustList = robust
        self.scalerDataList = [self.minmaxList, self.stdList, self.robustList]
        self.minmaxScaler = MinMaxScaler()
        self.stdScaler = StandardScaler()
        self.robustScaler = RobustScaler()
        self.scalerList = [self.minmaxScaler, self.stdScaler, self.robustScaler]

        #뉴런 구조
        self.lstm = nn.LSTM(self.inputSize, self.hiddenDim, self.layerNum, batch_first=True).to(self.device)
        self.hiddenLayer = nn.Linear(self.hiddenDim, self.outputSize).to(self.device)
        self.outputLayer = nn.ReLU().to(self.device)

    def sliceWindow(self, stock_data):
        data_raw = stock_data
        data = []

        for index in range(len(data_raw)-self.windowSize):
            data.append(data_raw[index: index+self.windowSize])

        data = np.array(data)
        return data

    def forward(self, x):
        h0 = torch.zeros(self.layerNum, x.size(0), self.hiddenDim).to(self.device).requires_grad_()

        c0 = torch.zeros(self.layerNum, x.size(0), self.hiddenDim).to(self.device).requires_grad_()

        out, (hn,cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.hiddenLayer(out[:, -1, :])
        out = self.outputLayer(out)
        return out

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
        slice_data_y = self.sliceWindow(data[self.output])
        x_slice_data = np.array(slice_data_x)
        y_slice_data = np.array(slice_data_y)

        return x_slice_data, y_slice_data


    def predict(self, model, data):
        x_slice, y_slice = self.dataProcessing(data)

        x_slice = x_slice[:,:-1]
        y_slice = np.array(y_slice[:,-1]).reshape(-1,1)

        x_predict_lstm = torch.from_numpy(x_slice).type(torch.Tensor).to(self.device)

        y_predict = model(x_predict_lstm)
        y_predict = self.minmaxScaler.inverse_transform(y_predict.detach().numpy())
        y_origin = self.sliceWindow(data[self.output])[:,-1]

        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()
        axes.plot(data[len(data)-len(y_slice):].index, y_origin, color='red', label='Real price')
        axes.plot(data[len(data)-len(y_slice):].index, y_predict, color='blue', label='Predict price')

        plt.xlabel('Time')
        plt.ylabel('price')
        plt.legend()
        plt.show()

    def learn(self, model, data):

        x_slice, y_slice = self.dataProcessing(data)
        total_size = len(x_slice)
        train_size = int(total_size*self.trainRate)
        x_train = x_slice[:train_size,:-1]
        x_test = x_slice[train_size:,:-1]

        y_train = np.array(y_slice[:train_size,-1]).reshape(-1,1)
        y_test = np.array(y_slice[train_size:,-1]).reshape(-1,1)

        print('training size is {}'.format(x_train.shape))
        print('test size is {}'.format(x_test.shape))

        x_train_lstm = torch.from_numpy(x_train).type(torch.Tensor).to(self.device)
        x_test_lstm = torch.from_numpy(x_test).type(torch.Tensor).to(self.device)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor).to(self.device)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor).to(self.device)

        loss_function = nn.BCELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(self.epochCnt)
        print(model)

        start_time = time.time()
        for t in range(self.epochCnt):

            y_train_pred = model(x_train_lstm)
            loss = loss_function(y_train_pred, y_train_lstm)
            if t % 5 == 0:
                print('Epoch ', t, 'MSE: ', loss.item())
            hist[t] = loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        train_time = time.time() - start_time
        print('Training Time : {}'.format(train_time))

        y_test_pred = model(x_test_lstm)

        y_test_pred = y_test_pred.to('cpu').detach().numpy()
        y_test_lstm = y_test_lstm.to('cpu').detach().numpy()
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        data.index = data['Date']
        axes.plot(data[len(data) - len(y_test_pred):].index, y_test_lstm, color='red', label='Real price')
        axes.plot(data[len(data) - len(y_test_pred):].index, y_test_pred, color='blue', label='Predict price')

        plt.xlabel('Time')
        plt.ylabel('price')
        plt.legend()
        plt.show()


    def save(self, model, filename='model.pt', mmScaler='mm.joblib'):
        torch.save(model.state_dict(), filename)
        joblib.dump(self.minmaxScaler, mmScaler)

    def load(self, filename='', mmScaler='mm.joblib'):
        model = torch.load(filename)
        self.minmaxScaler = joblib.load(mmScaler)
        return model

if __name__ == '__main__':
    #날짜를 정해서 불러오는 방법
    # start = (2020,1,1)
    # start = datetime.datetime(*start)
    # end = datetime.date.today()
    # df = pdr.DataReader('005930.KS', 'yahoo', start, end)

    df = pd.read_csv('samsung.csv')
    lstm = LSTMStock()
    lstm.run(lstm,df)
    lstm.save(lstm)