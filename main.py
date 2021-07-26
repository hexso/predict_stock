import FinanceDataReader as fdr
from LSTM import LSTMStock
from Stock import StockCal
import pandas as pd

if __name__ == '__main__':
    data = fdr.DataReader('005930')
    data.to_csv('stocks/samsung.csv')
    lstm = LSTMStock()
    stockCal = StockCal()
    df = pd.read_csv('stocks/samsung.csv')
    df = stockCal.getStockInput(df)
    df.to_csv('stocks/samsung.csv')

    lstm.setFile('stocks/samsung.csv', 'Date')
    
    #나스닥지수
    nq = fdr.DataReader('NASDAQCOM', data_source='fred')

    inputList = ['Open', 'Close', 'Volume', 'MACD','STOCHK','STOCHD',
                 'BUPPER','BMIDDLE','BLOWER','OBV','SMA20','SMA5']
    outputList = ['Close']

    lstm.setInput(inputList)
    lstm.setOutput(outputList)
    lstm.run(lstm)
