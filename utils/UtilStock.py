import talib
import pandas as pd

class StockCal:
    def __init__(self):
        pass


    def getStockInput(self, x, bollinger=20):
        x.columns = map(str.lower, x.columns)
        x.index = x['date']
        x['MACD'] = talib.MACD(x['close'])[0]
        x['STOCHK'], x['STOCHD'] = talib.STOCH(high=x['high'], low=x['low'], close=x['close'])
        x['BUPPER'], x['BMIDDLE'],x['BLOWER'] = talib.BBANDS(x['close'],bollinger)
        x['OBV'] = talib.OBV(x['close'],volume=x['volume'])
        x['SMA20'] = talib.SMA(x['close'],20)
        x['SMA5'] = talib.SMA(x['close'],5)
        x['RSI'] = talib.RSI(x['close'],20).fillna(100)
        x['OBVS'] = x['OBV'].ewm(20).mean() - x['OBV']
        x['VOLUME_CHANGE'] = talib.ROCP(x['volume'], timeperiod=1)
        x['HIGH_CHANGE'] = talib.ROC(x['high'], timeperiod=1)
        # start = x.iloc[0]['date']
        # end = x.iloc[-1]['date']
        # func = lambda x: 0 if x<0.05 else 1
        # x['CHANGE5'] = x['CHANGE5'].apply(func)
        # x['CHANGE5_TMW'] = x['CHANGE5'].shift(-1).fillna(0)
        # func = lambda x: 0 if x < 0.03 else 1
        # x['CHANGE3'] = x['change'].apply(func)
        # x['CHANGE3_TMW'] = x['CHANGE3'].shift(-1).fillna(0)
        # positive_func = lambda  x: 0 if x<0 else 1
        # x['UP'] = x['change'].apply(positive_func)
        # x['TMW_UP'] = x['change'].apply(positive_func).shift(-1).fillna(0)
        return x.fillna(0)

if __name__ == '__main__':
    df = pd.read_csv('../stocks/samsung.csv')
    stockCal = StockCal()
    data = stockCal.getStockInput(df)
    print(data)