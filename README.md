python main.py --simul --coin --mute --stock --stocksimul


--coinsimul : 시뮬레이션을 돌리는 옵션. coins.txt에 있는 리스트들의 코인들의 Data를 받는다.
          해당 Data들을 바탕으로 종료시 총 수익을 계산해준다.

--stocksimul : 시뮬레이션을 돌리는 옵션. stocks.txt에 있는 주식 Data를 받는다.
          해당 Data들을 바탕으로 종료시 총 수익을 계산해준다.

--coin : coins.txt의 코인들을 참고하여 data를 받는다.

--mute : 기본적으로 실행시 telegrame을 통해 메시지를 보낸다.
         해당 옵션시 메시지를 보내지 않는다.
         메시지의 경우 private.txt에 
        telegramtoken:
        telegramchatid:
        형식으로 써주어야 한다.

--stock : stock data들을 다운받아서 조건에 맞는 주식들을 추천한다.

