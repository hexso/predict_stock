'''
Data를 처리해서 Learning에 맞게 전달한다.
실시간 and excel형식의 데이터를 읽어서 전달.
'''
import os.path
from utils.upbit import UpbitTrade, GetCandle
import datetime as dt
import multiprocessing as mp

THREAD_CNT = 1
NOW = dt.datetime.now()
FROM_DATA = '2020-01-01'
SCRAPE_TIME_UNIT = 240
TIME_UNIT = {1: 'minute1', 3: 'minute3', 5: 'minute5', 10: 'minute10',
                          15: 'minute15', 30: 'minute30', 60: 'minute60', 240:'minute240',
                          1440:'day'}

def scrape_coin_data(coin_queue):
    while True:
        try:
            coin = coin_queue.get_nowait()
        except:
            print('Thread done')
            break

        try:
            coin_code = coin.strip()
            from_time = dt.datetime.strptime(FROM_DATA, "%Y-%m-%d")
            time_gap = NOW - from_time
            cnt = (time_gap.days * 24 * 60 + time_gap.seconds / 60) / SCRAPE_TIME_UNIT
            data = GetCandle(coin_code, unit=TIME_UNIT[SCRAPE_TIME_UNIT], count=cnt)
            coin_data = data.fillna(0)
            coin_data.columns = map(str.lower, coin_data.columns)
            coin_data.to_csv('coins/' + coin_code + '.csv')
            print("done {}".format(coin_code))
        except Exception as e:
            print("error {} {}".format(e, coin_code))

class CoinData:

    def __init__(self):
        self.trader = UpbitTrade()
        if not os.path.exists('coins'):
            os.makedirs('coins')

    #분단위로 Data를 받아온다.
    def GetFullData(self, coin_code='KRW-BTC', time_unit=240, output='excel', when='2021-11-12'):
        from_time = dt.datetime.strptime(when, "%Y-%m-%d")
        time_gap = NOW - from_time
        cnt = (time_gap.days*24*60 + time_gap.seconds/60)/time_unit
        print('코인 {} 총 {}개의 Data를 가져올 예정입니다. 소요시간 {}s 예정입니다.'.format(coin_code, cnt, cnt/1000))
        data = GetCandle(coin_code, unit=TIME_UNIT[time_unit], count=cnt)
        if output == 'excel':
            data.to_csv('coins/' + coin_code+'.csv')
        else:
            print('지원하지 않는 형식입니다.')
            exit(-1)
        return data

    def GetCoinLive(self, coin='KRW-BTC'):
        data = GetCandle(coin,unit='minute1',count=1)
        return data


    def download_all_coin_data(self):
        with open('coins.txt', encoding='cp949') as f:
            coins = f.readlines()

        manager = mp.Manager()
        coin_queue = manager.Queue()
        mps = []

        for coin in coins:
            coin_queue.put(coin)

        for process in range(THREAD_CNT):
            proc = mp.Process(target=scrape_coin_data, args=(coin_queue,))
            mps.append(proc)

        for proc in mps:
            proc.start()

        for proc in mps:
            proc.join()

def main():
    coin_data = CoinData()
    coin_data.download_all_coin_data()

if __name__ == '__main__':
    main()