import pandas as pd
import FinanceDataReader as fdr
from utils.UtilStock import StockCal
from datetime import datetime
import os
from threading import Thread

START_TIME = '2019'
START_DATE = '2019-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
STOCK_FOLDER = 'stocks'


class DataHandler:

    def __init__(self, log=True):
        self.total_data = []
        self.data_index = -1
        self.stock_index = -1
        self.stock_calculator = StockCal()
        self.stock_name = ''
        self.log = log

    def get_stocks_list(self, path='stocks'):
        self.stock_list = os.listdir(path)

    def load_data(self, path, start_time=START_DATE, end_time=END_DATE):
        try:
            self.total_data = pd.read_csv(path)
            self.total_data['Date'] = pd.to_datetime(self.total_data['Date'])
            self.total_data = self.stock_calculator.getStockInput(self.total_data)
            self.total_data = self.total_data[self.total_data['Date'].between(start_time, end_time)]
            if self.log is True:
                print("{} data is setted".format(path))
        except Exception as e:
            if self.log is True:
                print("load_data error {}".format(e))

    def set_next_stock(self, stock=None, start_time=START_DATE,end_time=END_DATE):
        if stock is not None:
            stock = stock + '.csv'
            self.stock_name = stock
        else:
            self.stock_index += 1
            if len(self.stock_list) <= self.stock_index:
                return False
            stock = self.stock_list[self.stock_index]
            self.stock_name = stock[:-4]
        self.load_data(STOCK_FOLDER+'/'+stock,start_time,end_time)
        self.data_index = -1

    def next_data(self):
        self.data_index += 1
        if len(self.total_data) is self.data_index:
            return 0
        self.total_data['Name'] = self.stock_name
        return self.total_data.iloc[self.data_index].to_dict()

    def download_stock_info(self):
        def do_thread(stocks, *args):
            for stock in stocks:
                try:
                    data = stock.split(':')
                    stock_data = fdr.DataReader(data[1].replace('\n', ''), START_TIME)
                    stock_data = stock_data.fillna(0)
                    stock_data['Change'] = round(stock_data['Change']*100, 2)
                    stock_data.to_csv('stocks/'+data[0]+'.csv')
                    print("done {} {}".format(data[0], data[1]))
                except Exception as e:
                    print("error {} {} {}".format(e, data[0], data[1]))

        with open('stocks.txt', encoding='cp949') as f:
            stocks = f.readlines()
            stock1 = stocks[:int(len(stocks)/2)]
            stock2 = stocks[int(len(stocks)/2):]
            print(stock1)
            print(stock2)
            th1 = Thread(target=do_thread, args=(stock1,))
            th2 = Thread(target=do_thread, args=(stock2,))
            th1.start()
            th2.start()
            th1.join()
            th2.join()

