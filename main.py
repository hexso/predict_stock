import FinanceDataReader as fdr
from LSTM import LSTMStock
from Stock import StockCal
import pandas as pd

if __name__ == '__main__':


    stock_code = {'samsung':'005930','신일전자':'002700'}
    # #주식코드로 데이터를 받아온다.
    for name, code in stock_code.items():
        data = fdr.DataReader(code)
        data.to_csv('stocks/' + name+'.csv')

    # #주식데이터로 보조지표를 만들어 낸다.
    filename = 'stocks/samsung.csv'
    stockCal = StockCal()
    df = pd.read_csv(filename)
    df = stockCal.getStockInput(df)
    df.to_csv(filename)

    data = pd.read_csv(filename)
    inputs = ['Change','Volume','NASDAQ']
    lstm = LSTMStock(minmax=inputs,output='Change',processor='cuda')
    lstm.learn(lstm,data)
    #lstm.save(lstm)
    # lstm.predict(lstm,data)
    # model = lstm.load('model.pt')
    # lstm.predict(lstm,data)
