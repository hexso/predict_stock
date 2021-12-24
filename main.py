from models.LSTM import LSTMStock
import pandas as pd

if __name__ == '__main__':


    stock_code ={}

    # #주식코드로 데이터를 받아온다.
    # with open('stocks.txt', 'r', encoding='cp949') as f:
    #     datas = f.readlines()
    #     for data in datas:
    #         data = data.strip('\n').split(':')
    #         stock_code[data[0]] = data[1]
    # for name, code in stock_code.items():
    #     data = fdr.DataReader(code)
    #     data.to_csv('stocks/' + name+'.csv')
    #
    # # #주식데이터로 보조지표를 만들어 낸다.
    # stockCal = StockCal()
    # for stock_name in stock_code.keys():
    #     print(stock_name)
    #     df = pd.read_csv('stocks/'+stock_name +'.csv')
    #     df = stockCal.getStockInput(df)
    #     df.to_csv('stocks/'+stock_name +'.csv')

    data = pd.read_csv('stocks/3S.csv')
    inputs = ['Change5','Volume','NASDAQ']
    # gru = GRUStock(origin=inputs, output='Change5',processor='cuda')
    # gru.learn(gru, data)
    lstm = LSTMStock(minmax=inputs, output='Change5',processor='cuda')
    lstm.learn(lstm,data)
    #lstm.save(lstm)
    # model = lstm.load('model.pt')
    # lstm.predict(lstm,data)
