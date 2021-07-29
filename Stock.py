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
        start = x.iloc[0]['Date']
        end = x.iloc[-1]['Date']
        nasdaq = pd.read_csv('NASDAQ.csv')
        nasdaq.index = nasdaq['DATE']
        x['NASDAQ'] = nasdaq.loc[start:end]['Change']
        func = lambda x: 0 if x<0.05 else 1
        x['Change5'] = x['Change'].apply(func)
        return x.fillna(0)

if __name__ == '__main__':
    df = pd.read_csv('stocks/samsung.csv')
    stockCal = StockCal()
    data = stockCal.getStockInput(df)
    print(data)