import talib
import pandas as pd

class StockCal:
    def __init__(self):
        pass


    def getStockInput(self, x, close='Close', bollinger=20, volume='Volume', high='High', low='Low'):
        x.index = x['Date']
        x['MACD'] = talib.MACD(x[close])[0]
        x['STOCHK'], x['STOCHD'] = talib.STOCH(high=x[high], low=x[low], close=x[close])
        x['BUPPER'], x['BMIDDLE'],x['BLOWER'] = talib.BBANDS(x[close],bollinger)
        x['OBV'] = talib.OBV(x[close],volume=x[volume])
        x['SMA20'] = talib.SMA(x[close],20)
        x['SMA5'] = talib.SMA(x[close],5)
        x['RSI'] = talib.RSI(x[close],20).fillna(100)
        x['OBVS'] = x['OBV'].ewm(20).mean() - x['OBV']
        start = x.iloc[0]['Date']
        end = x.iloc[-1]['Date']
        func = lambda x: 0 if x<0.05 else 1
        x['Change5'] = x['Change'].apply(func)
        x['Change5_tmw'] = x['Change5'].shift(-1).fillna(0)
        func = lambda x: 0 if x < 0.03 else 1
        x['Change3'] = x['Change'].apply(func)
        x['Change3_tmw'] = x['Change3'].shift(-1).fillna(0)
        positive_func = lambda  x: 0 if x<0 else 1
        x['up'] = x['Change'].apply(positive_func)
        x['tmw_up'] = x['Change'].apply(positive_func).shift(-1).fillna(0)
        return x.fillna(0)

if __name__ == '__main__':
    df = pd.read_csv('../stocks/samsung.csv')
    stockCal = StockCal()
    data = stockCal.getStockInput(df)
    print(data)