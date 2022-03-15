#-*- coding: utf-8 -*-
from utils.DataHandler import DataHandler
from datetime import datetime
from models.RSITrade import RSIAlgorithm
from models.VolumeChange import VolumeChange
from utils.telegram_bot import TelegramBot
from utils.CoinData import CoinData
from models.CoinTradeSimulator import CoinTradeSimulator
from models.StockTradeSimulator import StockTradeSimulator
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', action='store_true')
    parser.add_argument('--coin', action='store_true')
    parser.add_argument('--coinsimul', action='store_true')
    parser.add_argument('--stocksimul', action='store_true')
    parser.add_argument('--mute', action='store_true')
    args = parser.parse_args()

    if args.mute is True:
        tgBot = TelegramBot()

    if args.stock is True:
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

        result = str(rsi_stocks)
        for i in range(0,len(result),1000):
            tgBot.sendmsg(result[i:i+1000])

        result = str(volume_stocks)
        for i in range(0,len(result),1000):
            tgBot.sendmsg(result[i:i+1000])

    if args.coin is True:
        coin_data = CoinData()
        coin_data.download_all_coin_data()

    if args.coinsimul is True:
        with open('coins.txt', encoding='cp949') as f:
            stocks = f.readlines()
            for stock in stocks:
                stock = stock.strip()
                simulator = CoinTradeSimulator()
                simulator.set_data(coin=stock, time_stamp='2022-01-22')
                simulator.start('coin_simulator_output.txt')

    if args.stocksimul is True:
        with open('stocks.txt', encoding='cp949') as f:
            stocks = f.readlines()
            for stock in stocks:
                stock = stock.strip()
                data = stock.split(':')
                simulator = StockTradeSimulator()
                simulator.set_data(stock_code=data[1], start_time='2022-01-22')
                simulator.start('stock_simulator_output.txt')

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
