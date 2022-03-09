import pandas as pd
import FinanceDataReader as fdr
from utils.UtilStock import StockCal
from datetime import datetime
import os
from queue import Queue
import multiprocessing as mp

START_TIME = '2019'
START_DATE = '2019-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
STOCK_FOLDER = 'stocks'
THREAD_CNT = 3


def scrape_stock_data(stock_queue: Queue):
    while True:
        try:
            stock = stock_queue.get_nowait()
        except:
            print('Thread done')
            break

        try:
            data = stock.split(':')
            stock_data = fdr.DataReader(data[1].replace('\n', ''), START_TIME)
            stock_data = stock_data.fillna(0)
            stock_data.columns = map(str.lower, stock_data.columns)

            stock_data['change'] = round(stock_data['change'] * 100, 2)
            stock_data.to_csv('stocks/' + data[0] + '.csv')
            print("done {} {}".format(data[0], data[1]))
        except Exception as e:
            print("error {} {} {}".format(e, data[0], data[1]))

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
            self.total_data = self.total_data[self.total_data['date'].between(start_time, end_time)]
            if self.log is True:
                print("{} data is setted".format(path))
        except Exception as e:
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
        self.total_data['name'] = self.stock_name
        return self.total_data.iloc[self.data_index].to_dict()

    def download_stock_info(self):


        with open('stocks.txt', encoding='cp949') as f:
            stocks = f.readlines()

        manager = mp.Manager()
        stock_queue = manager.Queue()
        mps = []

        for stock in stocks:
            stock_queue.put(stock)

        for process in range(THREAD_CNT):
            proc = mp.Process(target=scrape_stock_data, args=(stock_queue,))
            mps.append(proc)

        for proc in mps:
            proc.start()

        for proc in mps:
            proc.join()

