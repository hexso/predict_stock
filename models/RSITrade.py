'''
RSI20, OBV osillator 20으로 알림
RSI가 30이하로 떨어지고, OBV가
'''

RSI_THRESHOLD = 30 # 30이하로 떨어지면 찾기
OBV_THRESHOLD = 0 #양수가 되면 찾기


class RSIAlgorithm:

    def __init__(self):
        pass

    def catch_stocks(self, datas):
        stock_list = list()
        for data in datas:
            if data['OBVS'] > 0 and data['RSI'] < 30:
                stock_list.append({"Name" : data['Name'], "OBVS" : data['OBVS'], "RSI" : data['RSI']})

        return stock_list





