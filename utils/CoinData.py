'''
Data를 처리해서 Learning에 맞게 전달한다.
실시간 and excel형식의 데이터를 읽어서 전달.
'''
from utils.upbit import UpbitTrade
import datetime as dt

class CoinData:

    def __init__(self):
        self.trader = UpbitTrade()


    #분단위로 Data를 받아온다.
    def GetFullData(self, stockcode='KRW-BTC', output='excel', when='2021-11-12 00:00:00'):

        now = dt.datetime.now()
        from_time = dt.datetime.strptime(when, "%Y-%m-%d %H:%M:%S")
        time_gap = now - from_time
        cnt = time_gap.days*60*24 + int(time_gap.seconds/60)
        print('코인 {} 총 {}개의 Data를 가져올 예정입니다. 소요시간 {}s 예정입니다.'.format(stockcode,cnt,cnt/1000))
        data = self.trader.GetCandle(stockcode, unit='minute1', count=cnt)
        if output == 'excel':
            data.to_csv('stocks/' + stockcode+'.csv')
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