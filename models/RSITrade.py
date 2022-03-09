'''
RSI20, OBV osillator 20으로 알림
RSI가 30이하로 떨어지고, OBV가 양수인 주식 데이터를 리턴해준다.
input : 여러개의 주식의 data가 list형식으로 들어가야 한다.
OBVS, RSI값이 있어야 한다.
'''

RSI_THRESHOLD = 30 # 30이하로 떨어지면 찾기
OBV_THRESHOLD = 0 #양수가 되면 찾기


class RSIAlgorithm:

    def __init__(self):
        pass

    def catch_stocks(self, datas: list):
        stock_list = list()
        for data in datas:
            if data['OBVS'] > 0 and data['RSI'] < 30:
                stock_list.append({"name" : data['name'], "OBVS" : data['OBVS'], "RSI" : data['RSI']})

        return stock_list





