'''
차트 보조지표들을 이용해서 시뮬레이션을 해본다.
'''

from utils.UtilStock import StockCal
from utils.DataHandler import DataHandler
START_BALANCE = 1000000
MFI_UPPER = 70
MFI_LOWER = 30
MFI_TRADE_FLAG = True
DI_TRADE_FLAG = True
BBAND_TRADE_FLAG = True




def underline(func):
    def deco(*args, **kwargs):
        print("=======================")
        func(*args, **kwargs)
        print("=======================")
    return deco


class StockTradeSimulator:
    def __init__(self):
        self.data_scraper = DataHandler()
        self.data_calculator = StockCal()
        self.balance = START_BALANCE
        self.account = []
        self.behavior_flag = 'buy'
        self.data = None

    def set_data(self, stock_code='005930', start_time='2022-01-22'):
        self.data = self.data_scraper.get_stock_info(stock_code, start_time)
        self.data = self.data_calculator.get_stock_indicators(self.data)
        self.data['stock_code'] = stock_code
        self.account.append({'stock_code': stock_code, 'amount': 0, 'avg_price': 0})

    def catch_buy_signal(self, data):
        if MFI_TRADE_FLAG is True:
            if data['MFI'] > MFI_LOWER:
                return False
        if DI_TRADE_FLAG is True:
            if data['MDI'] > data['ADX'] or data['MDI'] > data['PDI']:
                return False
        if BBAND_TRADE_FLAG is True:
            if data['low'] > data['LBBAND']:
                return False

        return True

    def catch_sell_signal(self, data):
        if MFI_TRADE_FLAG is True:
            if data['MFI'] > MFI_UPPER:
                print('MFI {}가 MFIUPPER : {}를 초과했습니다. 매도시점입니다.'.format(data['MFI'], MFI_UPPER))
                return True

        if DI_TRADE_FLAG is True:
            if data['MDI'] > data['ADX'] and data['MDI'] > data['PDI']:
                print('MDI : {}, PDI : {}, ADX : {}  매도시점입니다.'.format(data['MDI'], data['PDI'], data['ADX']))
                return True

        if BBAND_TRADE_FLAG is True:
            if data['UBBAND'] < data['high']:
                print('UBBAND {}를 high {}가 넘어섰습니다. 매도 시점입니다.'.format(data['UBBAND'], data['high']))
                return True

        return False

    @underline
    def buy_stock(self, data, amount=None):
        for stock_data in self.account:
            if stock_data['stock_code'] is data['stock_code']:
                buying_price = min(int((data['high'] + data['low'])/2), int(data['open']))

                if amount is None:
                    amount = int(self.balance/buying_price)

                if self.balance - (buying_price * amount) < 0:
                    print('구매하려는 양이 계좌잔고에 비해 많습니다.')
                    break

                self.balance = self.balance - (buying_price * amount)
                stock_data['avg_price'] = int(((stock_data['avg_price'] * stock_data['amount']) + (buying_price * amount))
                                             / (stock_data['amount'] + amount))
                stock_data['amount'] += amount
                print('매수가격 {} 매수량 {} '.format(buying_price, amount))
                break

    @underline
    def sell_stock(self, data, amount=None):
        for stock_data in self.account:
            if stock_data['stock_code'] is data['stock_code']:

                if amount is None:
                    amount = stock_data['amount']
                elif amount > stock_data['amount']:
                    print('판매 하려는 수량이 갖고 있는 수량에 비해 많습니다.')
                    break
                selling_price = max(int((data['high'] + data['low'])/2), int((data['open'] + data['high'])/2))
                profit_price = (selling_price * amount)-(stock_data['avg_price'] * amount)
                print("평단가는 {}, 판매가는 {} , 판매량 {} , 총 수익은 {}입니다."
                      .format(stock_data['avg_price'], selling_price, amount, profit_price))
                self.balance += selling_price * amount
                stock_data['amount'] -= amount
                if stock_data['amount'] == 0:
                    stock_data['avg_price'] = 0
                break

    def start(self, output=None):
        if self.data is None:
            print('set_data를 통해 data를 먼저 선택해줘야 합니다.')
            print("ex: set_data(stock='KRW-BTC',time_stamp='2022-01-22')")
            return False

        print('계좌 시작 잔고 : {}'.format(self.balance))

        for i in range(len(self.data)):
            stock_data = self.data.iloc[i]
            if self.behavior_flag is 'buy'\
                    and self.catch_buy_signal(stock_data) is True:
                print('매수시점입니다.')
                self.buy_stock(stock_data)
                self.behavior_flag = 'sell'

            if self.behavior_flag is 'sell'\
                    and self.catch_sell_signal(stock_data) is True:
                print('매도 시점입니다.')
                self.sell_stock(stock_data)
                self.behavior_flag = 'buy'


        print('시뮬레이터 후 계좌잔고는 {}, 총 수익률은 {}입니다.'.format(self.balance, round(self.balance/START_BALANCE*100, 2) - 100))
        if self.account[0]['amount'] > 0:
            balance = self.balance
            print('가지고 있는 코인')
            print(self.account)
            balance = balance + self.account[0]['amount'] * self.account[0]['avg_price']
            print('평단가로 계산한 계좌잔고는 {}, 총 수익률은 {}입니다.'.format(balance,
                                                           round(balance / START_BALANCE * 100, 2) - 100))

        if output is not None:
            with open(output, 'a') as f:
                f.write('주식코드 {} ,시뮬레이터 후 계좌잔고는 {}, 총 수익률은 {}입니다.\n'.format(self.data['stock_code'][0], self.balance, round(self.balance/START_BALANCE*100, 2) - 100))
                f.write(str(self.account) + '\n')
                f.write("=================================================================\n")

if __name__ == '__main__':
    simulator = StockTradeSimulator()
    simulator.set_data(stock_code='005930', start_time='2020-01-22')
    simulator.start('stock_simulator_output.txt')

