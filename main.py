import FinanceDataReader as fdr
from LSTM import LSTMStock
from Stock import StockCal
import pandas as pd

if __name__ == '__main__':

    filename = 'stocks/samsung.csv'
    #
    # #주식코드로 데이터를 받아온다.
    # data = fdr.DataReader('005930')
    # data.to_csv(filename)
    #
    # #주식데이터로 보조지표를 만들어 낸다.
    # stockCal = StockCal()
    # df = pd.read_csv(filename)
    # df = stockCal.getStockInput(df)
    # df.to_csv(filename)
    #
    # # 나스닥지수
    # # nq = fdr.DataReader('NASDAQCOM', data_source='fred')
    data = pd.read_csv(filename)
    lstm = LSTMStock()
    lstm.learn(lstm,data)
    lstm.save(lstm)
    # lstm.predict(lstm,data)
    # model = lstm.load('model.pt')
    # lstm.predict(lstm,data)
