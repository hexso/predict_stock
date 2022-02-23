import pandas as pd
import FinanceDataReader as fdr

START_TIME = '2019'

class DataHandler:

    def __init__(self):
        self.total_data = []
        self.index = -1

    def load_data(self, path):
        try:
            self.total_data = pd.read_csv(path)
            print("{} data is setted")
        except:
            print("error load_data")

    def next_data(self):
        self.index += 1
        if len(self.total_data) is self.index:
            return 0
        return self.total_data[self.index]

    def download_stock_info(self):
        with open('stocks.txt') as f:
            stocks = f.readlines()
            for stock in stocks:
                try:
                    data = stock.split(':')
                    stock_data = fdr.DataReader(data[1].replace('\n',''), START_TIME)
                    stock_data['Change'] = round(stock_data['Change']*100, 2)
                    stock_data.to_csv('stocks/'+data[0]+'.csv')
                    print("done {} {}".format(data[0], data[1]))
                except:
                    print("error {} {}".format(data[0], data[1]))

