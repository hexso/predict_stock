'''
라오어의 무한매수법
3배 레버리지를 매일매일 일정 금액만큼 산다.
최대 40번을 사고 수익률이 10%이상일 경우 무조건 매도한다.

'''
from utils.DataHandler import DataHandler


TOTAL_BALANCE = 8000000
MOMENT_PRICE = 200000
MAXIMUM_COUNT = 40
SELL_PROFIT_RATE = 10


class LaoInfiniteTrade:

    def __init__(self):
        self.stock_balance = {}
        self.data = None
        self.stock_name = None
        self.balance = TOTAL_BALANCE
        self.datahandler = DataHandler()
        self.total_profit = 0
    def set_data(self, path):
        self.data = self.datahandler.load_data(path)
        self.stock_name = path.split('/')[-1]
        stock_data = {'name':self.stock_name, 'amount':0, 'avg_price':0}
        self.stock_balance=stock_data

    def cal_profit_rate(self, now_price):
        if self.stock_balance['avg_price'] == 0:
            return 0
        return (now_price/self.stock_balance['avg_price'])*100 -100

    def start(self):
        if self.data is None:
            print('set_data를 통해 data를 먼저 선택해줘야 합니다.')
        buy_cnt = 0
        for index, data in self.data.iterrows():
            profit_rate = self.cal_profit_rate(data['open'])
            if profit_rate > SELL_PROFIT_RATE or buy_cnt >= MAXIMUM_COUNT:
                profit = self.stock_balance['amount']*data['open']
                self.balance += profit
                self.total_profit += profit
                self.stock_balance['amount'] = 0
                self.stock_balance['avg_price'] = 0
                print('이익률이 {}를 초과하여 매도했습니다. 수익률: {}'.format(SELL_PROFIT_RATE, profit_rate))
                print('현재 잔고는 {} 입니다.'.format(self.balance))
                buy_cnt = 0
                continue
            buy_amount = int(MOMENT_PRICE/data['open'])
            self.balance -= (buy_amount*data['open'])
            avg_price = ((self.stock_balance['avg_price']*self.stock_balance['amount'])
                         + (buy_amount*data['open']))/(self.stock_balance['amount']+buy_amount)
            self.stock_balance['amount'] += buy_amount
            self.stock_balance['avg_price'] = avg_price
            buy_cnt += 1




if __name__ == '__main__':
    simulator = LaoInfiniteTrade()
    simulator.set_data('../stocks/FNGU.csv')
    simulator.start()
    print(simulator.total_profit)


