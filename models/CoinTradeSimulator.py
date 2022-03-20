'''
코인 차트 보조지표들을 이용해서 시뮬레이션을 해본다.
'''

from utils.CoinData import CoinData
from utils.UtilStock import StockCal

START_BALANCE = 1000000
MFI_UPPER = 70
MFI_LOWER = 30


class CoinTradeSimulator:
    def __init__(self):
        self.data_scraper = CoinData()
        self.data_calculator = StockCal()
        self.balance = START_BALANCE
        self.account = []
        self.behavior_flag = 'buy'
        self.data = None

    def set_data(self, coin='KRW-BTC', time_stamp='2022-01-22', time_unit=240):
        self.data = self.data_scraper.GetFullData(coin, when=time_stamp, time_unit=time_unit)
        self.data = self.data_calculator.get_stock_indicators(self.data)
        self.data['coin'] = coin
        self.account.append({'coin':coin, 'amount':0, 'avg_price':0})

    def catch_buy_signal(self, data):
        if data['MFI'] > MFI_LOWER:
            return False

        if data['MDI'] > data['ADX'] or data['MDI'] > data['PDI']:
            return False

        if data['low'] > data['LBBAND']:
            return False

        return True

    def catch_sell_signal(self, data):
        if data['MFI'] > MFI_UPPER:
            print('MFI {}가 MFIUPPER : {}를 초과했습니다. 매도시점입니다.'.format(data['MFI'], MFI_UPPER))
            return True

        if data['MDI'] > data['ADX'] and data['MDI'] > data['PDI']:
            print('MDI : {}, PDI : {}, ADX : {}  매도시점입니다.'.format(data['MDI'], data['PDI'], data['ADX']))
            return True

        if data['UBBAND'] < data['high']:
            print('UBBAND {}를 high {}가 넘어섰습니다. 매도 시점입니다.'.format(data['UBBAND'], data['high']))
            return True

        return False

    def buy_coin(self, data, amount=None):
        for coin_data in self.account:
            if coin_data['coin'] is data['coin']:
                buying_price = min(int((data['high'] + data['low'])/2), int(data['open']))

                #코인의 경우에는 소수점단위로 구매가 가능하다.
                if amount is None:
                    amount = round(self.balance/buying_price, 5)

                if self.balance - (buying_price * amount) < 0:
                    print('구매하려는 양이 계좌잔고에 비해 많습니다.')
                    break

                self.balance = self.balance - (buying_price * amount)
                coin_data['avg_price'] = int(((coin_data['avg_price'] * coin_data['amount']) + (buying_price * amount))
                                             / (coin_data['amount'] + amount))
                coin_data['amount'] += amount
                print(amount)
                print(self.balance)
                print(buying_price)
                break

    def sell_coin(self, data, amount=None):
        for coin_data in self.account:
            if coin_data['coin'] is data['coin']:

                if amount is None:
                    amount = coin_data['amount']
                elif amount > coin_data['amount']:
                    print('판매 하려는 수량이 갖고 있는 수량에 비해 많습니다.')
                    break
                selling_price = max(int((data['high'] + data['low'])/2), int((data['open'] + data['high'])/2))
                profit_price = (selling_price * amount)-(coin_data['avg_price'] * amount)
                print("평단가는 {}, 판매가는 {} , 판매량 {} , 총 수익은 {}입니다."
                      .format(coin_data['avg_price'], selling_price, amount, profit_price))
                self.balance += selling_price * amount
                coin_data['amount'] -= amount
                if coin_data['amount'] == 0:
                    coin_data['avg_price'] = 0
                break

    def start(self, output=None):
        if self.data is None:
            print('set_data를 통해 data를 먼저 선택해줘야 합니다.')
            print("ex: set_data(coin='KRW-BTC',time_stamp='2022-01-22')")
            return False

        print('계좌 시작 잔고 : {}'.format(self.balance))

        for i in range(len(self.data)):
            coin_data = self.data.iloc[i]
            if self.behavior_flag is 'buy'\
                    and self.catch_buy_signal(coin_data) is True:
                print('매수시점입니다.')
                self.buy_coin(coin_data)
                self.behavior_flag = 'sell'

            if self.behavior_flag is 'sell'\
                    and self.catch_sell_signal(coin_data) is True:
                print('매도 시점입니다.')
                self.sell_coin(coin_data)
                self.behavior_flag = 'buy'


        print('시뮬레이터 후 계좌잔고는 {}, 총 수익률은 {}입니다.'.format(self.balance, round(self.balance/START_BALANCE*100, 2) - 100))
        print('가지고 있는 코인')
        print(self.account)
        if output is not None:
            with open(output, 'a') as f:
                f.write('시뮬레이터 후 계좌잔고는 {}, 총 수익률은 {}입니다.\n'.format(self.balance, round(self.balance/START_BALANCE*100, 2) - 100))
                f.write(str(self.account) + '\n')

if __name__ == '__main__':
    simulator = CoinTradeSimulator()
    simulator.set_data(coin='KRW-BTC', time_stamp='2022-01-22')
    simulator.start('coin_simulator_output.txt')

