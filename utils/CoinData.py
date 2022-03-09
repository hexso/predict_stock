'''
Data를 처리해서 Learning에 맞게 전달한다.
실시간 and excel형식의 데이터를 읽어서 전달.
'''
import os.path

from utils.upbit import UpbitTrade
import datetime as dt

class CoinData:

    def __init__(self):
        self.trader = UpbitTrade()
        self.time_unit = {1: 'minute1', 3: 'minute3', 5: 'minute5', 10: 'minute10',
                          15: 'minute15', 30: 'minute30', 60: 'minute60', 240:'minute240',
                          1440:'day'}
        if not os.path.exists('coins'):
            os.makedirs('coins')

    #분단위로 Data를 받아온다.
    def GetFullData(self, stockcode='KRW-BTC', time_unit=240, output='excel', when='2021-11-12'):
        now = dt.datetime.now()
        from_time = dt.datetime.strptime(when, "%Y-%m-%d")
        time_gap = now - from_time
        cnt = (time_gap.days*24*60 + time_gap.seconds/60)/time_unit
        print('코인 {} 총 {}개의 Data를 가져올 예정입니다. 소요시간 {}s 예정입니다.'.format(stockcode,cnt,cnt/1000))
        data = self.trader.GetCandle(stockcode, unit=self.time_unit[time_unit], count=cnt)
        if output == 'excel':
            data.to_csv('coins/' + stockcode+'.csv')
        else:
            print('지원하지 않는 형식입니다.')
            exit(-1)
        return data

    def GetCoinLive(self, coin='KRW-BTC'):
        data = self.trader.GetCandle(coin,unit='minute1',count=1)
        return data


if __name__ == '__main__':
    coin_data = CoinData()
    coin_data.GetFullData()