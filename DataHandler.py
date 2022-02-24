import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
import os

START_TIME = '2019'
START_DATE = '2019-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
STOCK_FOLDER = 'stocks'

class DataHandler:

    def __init__(self):
        self.total_data = []
        self.data_index = -1
        self.stock_index = -1

    def get_stocks_list(self, path='stocks'):
        self.stock_list = os.listdir(path)

    def load_data(self, path, start_time=START_DATE, end_time=END_DATE):
        try:
            self.total_data = pd.read_csv(path)
            self.total_data['Date'] = pd.to_datetime(self.total_data['Date'])
            self.total_data[self.total_data['Date'].between(start_time, end_time)]
            print("{} data is setted".format(path))
        except:
            print("error load_data")

    def set_next_stock(self, stock=None):
        if stock is not None:
            stock = stock + '.csv'
        else:
            self.stock_index += 1
            stock = self.stock_list[self.stock_index]
        self.load_data(STOCK_FOLDER+'/'+stock)

    def next_data(self):
        self.data_index += 1
        if len(self.total_data) is self.data_index:
            return 0
        return self.total_data.loc[self.data_index].to_dict()

    def download_stock_info(self):
        with open('stocks.txt') as f:
            stocks = f.readlines()
            for stock in stocks:
                try:
                    data = stock.split(':')
                    stock_data = fdr.DataReader(data[1].replace('\n',''), START_TIME)
                    stock_data = stock_data.fillna(0)
                    stock_data['Change'] = round(stock_data['Change']*100, 2)
                    stock_data.to_csv('stocks/'+data[0]+'.csv')
                    print("done {} {}".format(data[0], data[1]))
                except Exception as e:
                    print("error {} {} {}".format(e, data[0], data[1]))

