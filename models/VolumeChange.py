'''
종가쯤에 전일대비 거래량 10배이상 증가했고
등락률이 -5~5 사이인 주식을 찾는다.
input : 여러개의 주식의 data가 list형식으로 들어가야 한다.
VOLUME_CHANGE: 거래량 변화율
High_Change: 고가 변화율

'''

VOLUME_CHANGE_RATE = 10
CHANGE_RATE = 7
TOTAL_TRADE_AMOUNT = 100000000


class VolumeChange:

    def __init__(self):
        pass

    def catch_stocks(self, datas: list):
        stock_list = list()
        for data in datas:
            if data['VOLUME_CHANGE'] > VOLUME_CHANGE_RATE and \
                    data['High_Change'] < CHANGE_RATE and data['High_Change'] > -CHANGE_RATE and \
                data['Volume']*data['Open'] > TOTAL_TRADE_AMOUNT:
                stock_list.append({"Name" : data['Name'], "VOLUME_CHANGE" : data['VOLUME_CHANGE'], "High_Change" : data['High_Change']})

        return stock_list





