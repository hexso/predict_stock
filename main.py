from DataHandler import DataHandler
from datetime import datetime
from models.RSITrade import RSIAlgorithm
from models.VolumeChange import VolumeChange
from utils.telegram_bot import TelegramBot

if __name__ == '__main__':
    data_handler = DataHandler(False)
    data_handler.download_stock_info()
    data_handler.get_stocks_list()
    today_date = datetime.now().strftime('%Y-%m-%d')
    today_stock_data = list()
    i = 0
    while 1:
        if data_handler.set_next_stock(start_time=today_date) is False:
            break
        data = data_handler.next_data()
        if data is not 0:
            today_stock_data.append(data)
        i += 1
        if i % 100 is 0:
            print(i)
    algorithm = RSIAlgorithm()
    algorithm2 = VolumeChange()
    rsi_stocks = algorithm.catch_stocks(today_stock_data)
    volume_stocks = algorithm2.catch_stocks(today_stock_data)
    with open('outputs/'+today_date+'_output.txt','w') as f:
        for stock in rsi_stocks:
            value = [str(i) for i in stock.values()]
            f.writelines("_".join(value))
            f.write('\n')
        f.write('====================================================')
        for stock in volume_stocks:
            value = [str(i) for i in stock.values()]
            f.writelines("_".join(value))
            f.write('\n')

    print('done')

    tgBot = TelegramBot()

    result = str(rsi_stocks)
    for i in range(0,len(result),1000):
        tgBot.sendmsg(result[i:i+1000])

    result = str(volume_stocks)
    for i in range(0,len(result),1000):
        tgBot.sendmsg(result[i:i+1000])

'''
    # # #주식데이터로 보조지표를 만들어 낸다.
    # stockCal = StockCal()
    # for stock_name in stock_code.keys():
    #     print(stock_name)
    #     df = pd.read_csv('stocks/'+stock_name +'.csv')
    #     df = stockCal.getStockInput(df)
    #     df.to_csv('stocks/'+stock_name +'.csv')

    # data = pd.read_csv('stocks/3S.csv')
    # inputs = ['Change5','Volume','NASDAQ']
    # gru = GRUStock(origin=inputs, output='Change5',processor='cuda')
    # gru.learn(gru, data)
    # lstm = LSTMStock(minmax=inputs, output='Change5',processor='cuda')
    # lstm.learn(lstm,data)
    #lstm.save(lstm)
    # model = lstm.load('model.pt')
    # lstm.predict(lstm,data)
'''
