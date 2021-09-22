import gym
from gym import spaces
import numpy as np

ACCOUNT_BALANCE = 10000000
MAX_TIME_STAMP = 10000
LOOKBACK_WINDOW_SIZE = 60
MAX_DIVIDE = 2147483647


class StockTradingEnv(gym.Env):

    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.time_stamp = 0
        self.balance = ACCOUNT_BALANCE
        self.stock_amount = 0
        self.avg_price = 0
        self.total_budget = ACCOUNT_BALANCE

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, LOOKBACK_WINDOW_SIZE + 2), dtype=np.float16)
        self.history = []

    def reset(self):

        self.time_stamp = 0
        self.balance = ACCOUNT_BALANCE
        self.stock_amount = 0
        self.avg_price = 0
        self.history = []
        self.total_budget = ACCOUNT_BALANCE

        return self.getState()


    def trade(self, action):

        open_price = self.stock_data.loc[self.time_stamp, 'Open']
        close_price = self.stock_data.loc[self.time_stamp, 'Close']

        action_type = action[0]
        amount = action[1]

        # 주식을 사는 행위
        if action_type < 1:
            possible_amount = int(self.balance / open_price)
            prev_cost = self.stock_amount * self.avg_price

            # 가능한 양에서 일정만큼만 산다.
            buying_amount = int(possible_amount * amount)
            buying_cost = buying_amount * open_price

            self.balance -= buying_cost
            if self.stock_amount + buying_amount >0:
                self.avg_price = (prev_cost + buying_cost) / (self.stock_amount + buying_amount)
            else:
                self.avg_price = 0

            self.stock_amount += buying_amount

            # 거래기록 저장
            if buying_amount > 0:
                self.history.append({'step': self.time_stamp,
                                     'amount': buying_amount,
                                     'cost': buying_cost,
                                     'type': 'buy',
                                     'total': self.balance + self.avg_price * self.stock_amount})



        # 주식을 파는 행위
        elif action_type < 2:
            selling_amount = int(self.stock_amount * amount)

            self.stock_amount -= selling_amount
            self.balance += selling_amount * open_price

            # 거래기록 저장
            if selling_amount > 0:
                self.history.append({'step': self.time_stamp,
                                     'amount': selling_amount,
                                     'cost': open_price,
                                     'type': 'sell',
                                     'total': self.balance + self.avg_price * self.stock_amount})

        # 유지하기
        else:
            pass

        self.total_budget = self.balance + self.avg_price * self.stock_amount

        # 주식이 하나도 없을 경우 평단가 0으로 초기화
        if self.stock_amount == 0:
            self.avg_price = 0

    def getState(self):
        frame = np.zeros((5, LOOKBACK_WINDOW_SIZE + 1))
        # Get the stock data points for the last 5 days and scale to between 0-1
        np.put(frame, [0, 4], [
            self.stock_data.loc[self.time_stamp: self.time_stamp +
                                           LOOKBACK_WINDOW_SIZE, 'Open'].values / MAX_DIVIDE,
            self.stock_data.loc[self.time_stamp: self.time_stamp +
                                           LOOKBACK_WINDOW_SIZE, 'High'].values / MAX_DIVIDE,
            self.stock_data.loc[self.time_stamp: self.time_stamp +
                                           LOOKBACK_WINDOW_SIZE, 'Low'].values / MAX_DIVIDE,
            self.stock_data.loc[self.time_stamp: self.time_stamp +
                                           LOOKBACK_WINDOW_SIZE, 'Close'].values / MAX_DIVIDE,
            self.stock_data.loc[self.time_stamp: self.time_stamp +
                                           LOOKBACK_WINDOW_SIZE, 'Volume'].values / MAX_DIVIDE,
        ])

        state = np.append(frame, [
            [self.balance / MAX_DIVIDE],
            [self.time_stamp / MAX_DIVIDE],
            [self.stock_amount / MAX_DIVIDE],
            [self.avg_price / MAX_DIVIDE],
            [self.total_budget / MAX_DIVIDE],
        ], axis=1)

        return state

    def step(self, action):

        self.trade(action)
        self.time_stamp += 1

        delay_modifier = (self.time_stamp / MAX_TIME_STAMP)


        state = self.getState()
        reward = self.total_budget * delay_modifier
        done = self.total_budget < 0 or self.time_stamp >= len(self.stock_data.loc[:,'Open'].values)

        return state, reward, done, {}

    def render(self, mode="human"):
        print(self.history)

    def close(self):
        pass

if __name__ == '__main__':
    import pandas as pd
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
    df = pd.read_csv('stocks/samsung.csv')
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose= 1, device="cuda")
    model.learn(total_timesteps=200000)
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()