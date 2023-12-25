# 最简单微电网。电池+负载+光伏
# 增加电池奖励，改初始电池状态
# 增加EV
# reward = reward
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import gym
import gym.spaces as spaces
# # 温度
# DEFAULT_OUTDOOR_TEMPERATURE = np.genfromtxt("PV_MG1.csv", delimiter=',',skip_header=1, usecols=[1])
# # generation
# DEFAULT_POWER_GENERATED = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
# # load
# DEFAULT_BASE_LOAD = np.genfromtxt("load_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
global DEFAULT_OUTDOOR_TEMPERATURE_old
global DEFAULT_POWER_GENERATED_old
global DEFAULT_BASE_LOAD_old
global DEFAULT_OUTDOOR_TEMPERATURE
global DEFAULT_POWER_GENERATED
global DEFAULT_BASE_LOAD
# 温度
DEFAULT_OUTDOOR_TEMPERATURE_old = np.genfromtxt("PV_MG1_G.csv", delimiter=',',skip_header=0, usecols=[1])
# generation
DEFAULT_POWER_GENERATED_old = np.genfromtxt("PV_MG1_G.csv", delimiter=',', skip_header=0, usecols=[-1])
# load
DEFAULT_BASE_LOAD_old = np.genfromtxt("load_MG1_G.csv", delimiter=',', skip_header=0, usecols=[-1])
# print("DEFAULT_OUTDOOR_TEMPERATURE 1",DEFAULT_OUTDOOR_TEMPERATURE )
global k
k = 1
# DEFAULT_OUTDOOR_TEMPERATURE=DEFAULT_OUTDOOR_TEMPERATURE[0:97]
# DEFAULT_POWER_GENERATED=DEFAULT_POWER_GENERATED[0:97]
# DEFAULT_BASE_LOAD = DEFAULT_BASE_LOAD[0:97]
DEFAULT_OUTDOOR_TEMPERATURE=DEFAULT_OUTDOOR_TEMPERATURE_old[96*k:96*k+97]
DEFAULT_POWER_GENERATED=DEFAULT_POWER_GENERATED_old[96*k:96*k+97]
DEFAULT_BASE_LOAD = DEFAULT_BASE_LOAD_old[96*k:96*k+97]
# print("k",k)
# print("DEFAULT_OUTDOOR_TEMPERATURE 2",DEFAULT_OUTDOOR_TEMPERATURE )
# price
DEFAULT_MARKET_PRICE = np.genfromtxt("buy_price.csv", delimiter=',', skip_header=1, usecols=[-1])
# Length of one episode
DEFAULT_ITERATIONS = 96
# Battery characteristics (kwh)
DEFAULT_BAT_CAPACITY = 10
DEFAULT_MAX_DISCHARGE = 2
MAX_R = 100
ACTIONS = [-1, 1]

""" EVs"""
class EV:
    def __init__(self,time_interval):
        self.interval = time_interval
        # self.t = t
        # self.energy_EV = 0
    def current_energy_EV(self,power_EV,t_EV_a,t_EV_d,energy_EV_init):
        # EV1
        if self.time + 1 == t_EV_a:
            self.energy_EV = energy_EV_init
        # elif t_EV_a < self.time + 1 <= t_EV_d:
        #     self.energy_EV = self.energy_EV - power_EV*self.interval
        elif t_EV_a < self.time + 1 <= t_EV_d and self.energy_EV>0: # 添加EV限制条件
            self.energy_EV = self.energy_EV - power_EV*self.interval
            if self.energy_EV>0:
                self.energy_EV = self.energy_EV
            else:
                self.energy_EV = 0
        else:
            self.energy_EV = 0
        return self.energy_EV
    def set_time(self, time):
        self.time = time

"""ACs"""
class AC:
    def __init__(self, time_interval, Temperature_out):
        self.interval = time_interval
        self.Temperature_out = Temperature_out
        # self.T_AC1 = self.T_out
        # self.S_AC1 = 0
        # self.cT = 25
    def current_S_AC1(self, t_AC_e, t_AC_l, T_AC_min, T_AC_max):
        # if self.time ==0: print("Temperature_out",self.Temperature_out.current_temperature_out(self.time))
        # if t_AC_e <= self.time+1 <= t_AC_l and self.T_AC1 < 25:
        # if t_AC_e <= self.time+1 <= t_AC_l and (self.T_AC1<T_AC_min or self.T_AC1>T_AC_max):
        if t_AC_e <= self.time + 1 <= t_AC_l:
            self.S_AC1 = 1
        else:
            self.S_AC1 = 0
        return self.S_AC1
    def current_T_AC1(self,power_AC,a_AC,b_AC):
        #AC1
        # print("1",self.T_AC1)
        # print("2",self.Temperature_out.current_temperature_out(self.time))
        # self.T_AC1 = self.T_AC1 + (1-(math.exp(-a_AC*self.interval)))*(self.Temperature_out.current_temperature_out(self.time)-self.T_AC1-b_AC*power_AC)
        self.T_AC1 = self.T_AC1 + (1 - (math.exp(-a_AC * self.interval))) * (self.Temperature_out.current_temperature_out(self.time)-self.T_AC1  + b_AC * power_AC)
        return  self.T_AC1

    def set_time(self,time):
        self.time = time

    def reset(self, time):
        self.T_AC1 = self.Temperature_out.current_temperature_out(time)

"""Tempreture_out"""
class Temperature:
    def __init__(self, temperature_out):
        self.T_out= temperature_out

    def current_temperature_out(self, time):
        return self.T_out[time]

"""BATTERY"""
class Battery:
    def __init__(self, capacity, rateB):
        self.rateB = rateB  # 电池充放电效率
        # self.a = action  # 电池动作
        # self.power_PV = power_PV  # 光伏发电功率
        # self.timeB = timeB  # 充放电时间
        self.capacity = capacity  # 电池容量
        # self.RC = 0 # remaining capacity 剩电

    def charge(self, E):  # 充电 输入：充电电量 输出：多余电量
        empty = self.capacity - self.RC  # 满电 - 剩电
        if empty <= 0:
            return E
        else:
            self.RC += self.rateB * E  # 剩电 = 剩电 + 充电速率 * E
            leftover = self.RC - self.capacity
            self.RC = min(self.capacity, self.RC)  # 剩电 = min(满电，剩电)
            return max(leftover/self.rateB, 0)  # 返回 max(多余电，0）

    def supply(self, E):  # 供应（放电） 输入：需要供应电量 输出：实际能供应的电量
        remaining = self.RC  # 剩电
        self.RC -= min(E/self.rateB, remaining)  # 剩电 = 剩电 - min(E,剩电)
        self.RC = max(self.RC, 0)  # 剩电 = max(剩电,0)
        return min(E, remaining*self.rateB)   # 返回 min(E,剩电)
        # return min(E, remaining)   # 返回 min(E,剩电)
    @property
    def SoC(self):
        return self.RC / self.capacity  # 荷电状态变量
    @property
    def rc(self):
        return self.RC

    def reset(self):
        # self.RC = 0
        self.RC = self.capacity * 0.85

"""GRID"""
class Grid:  # 微电网
    def __init__(self, buy_price):
        self.buy_price = buy_price
        # self.time = 0

    def sell(self, E):  # 向外网售电
        return self.buy_price.current_price(self.time) * 0.8 * E

    def buy(self, E):  # 向外网购电
        # return self.buy_price.current_price[self.time] * E
        return self.buy_price.current_price(self.time) * E

    def set_time(self, time):
        self.time = time

    def total_cost(self, prices, energy):
        return sum(prices * energy)

"""PV"""
class Generation:
    def __init__(self, generation):
        self.power = generation

    def current_generation(self, time):
        return self.power[time]

"""LOAD"""
class Load:
    def __init__(self, load):
        self.load = load

    def current_load(self, time):
        return self.load[time]

"""PRICE"""
class Price:
    def __init__(self, price):
        # print("price", price)
        self.price = price

    def current_price(self, time):
        # print("time", time)
        return self.price[time]

"""MG"""
class MicroGridEnv(gym.Env):
    def __init__(self, **kwargs):  # 全局变量的传参
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.bat_capacity = kwargs.get("battery_capacity", DEFAULT_BAT_CAPACITY)
        self.max_discharge = kwargs.get("max_discharge", DEFAULT_MAX_DISCHARGE)
        self.day = 0
        # The current timestep
        # self.time_step = 0
        self.interval = 0.25  # 15min
        # print(DEFAULT_MARKET_PRICE.shape)
        self.buy_price = Price(kwargs.get("normal_price", DEFAULT_MARKET_PRICE))
        # self.sell_price = self.buy_price * 0.8
        self.generation = Generation(kwargs.get("generation_data", DEFAULT_POWER_GENERATED))
        self.load = Load(kwargs.get("load_data", DEFAULT_BASE_LOAD))
        self.grid = Grid(buy_price=self.buy_price)
        self.battery = Battery(capacity=self.bat_capacity, rateB=0.9)
        self.review_SoC = []

        self.EV = EV(time_interval=self.interval)
        self.temperature_out = Temperature(kwargs.get("temperature_data", DEFAULT_OUTDOOR_TEMPERATURE))
        self.AC = AC(time_interval=self.interval, Temperature_out=self.temperature_out)

        # SPACE
        # self.action_space_sep = spaces.Box(low=0, high=1, dtype=np.float32,
        #                                shape=(13,))
        self.action_space = spaces.Box(low = -1,high = 1,shape = (1, ))  # 动作空间 80

        # Observations: A vector of TCLs SoCs + loads +battery soc+ power generation + price + temperature + time of day
        self.observation_space = spaces.Box(low=-100, high=100, dtype=np.float32,
                                            shape=(3,))



    def _build_state(self):
        """
        将当前状态表示形式为一个向量。
        返回：一维状态矢量，包含负载，当前电池SOC，当前发电
        """
        # print("self.day ,self.iterations ,self.time_step", self.day ,self.iterations , self.time_step)
        current_generation = self.generation.current_generation(self.day * self.iterations + self.time_step)
        current_load = self.load.current_load(self.day * self.iterations + self.time_step)
        current_SOC = self.battery.SoC
        # current_SoC = np.float(current_SoC)
        state = np.array([current_generation, current_load, current_SOC])
        # print(state)
        # print("current_generation",current_generation)
        # print("current_load",current_load)
        # print("current_soc",current_SoC)
        # print("review_SoC",self.review_SoC)
        # print("SoC", SoC)
        return state

    def _build_info(self):
        """
        返回要按状态给出的杂项信息的字典。
        在这里，这意味着提供对未来的预测
        价格和温度（接下来的24小时）
        """
        return {"forecast_times": np.arange(0, self.iterations)}

    def step(self, action):
        """
        参数：
            操作：列表。

        退货：
            状态： 当前状态
            奖励：最后一次操作获得了多少奖励
            终端：游戏是否结束的布尔值（最大迭代次数）
            信息：无（此处未使用）
        """
        self.grid.set_time(self.day * self.iterations + self.time_step)
        self.EV.set_time(self.day * self.iterations + self.time_step)
        self.AC.set_time(self.day * self.iterations + self.time_step)
        reward = 0
        # reward_1 = 0
        # reward_2 = 0
        cost_grid = 0
        a_BES = action[0]
        a_BES = 2 * (a_BES - 0.5)
        self.action_BES = a_BES
        # EV1
        power_EV1 = action[1]
        # power_EV1 = 0.5 * power_EV1 + 0.5  # [-1,1]到[0,1]
        power_EV1 = power_EV1 * 3.4  # [0,1]到[0,3.4]
        # power_EV1 = power_EV1 * 6.8  # [0,1]到[0,3.4]
        # AC1
        power_AC1 = action[2]
        # power_AC1 = action[2] * 2
        # power_AC1 = 0.5 * power_AC1 + 0.5  # [-1,1]到[0,1]

        # Update state according to action
        # print("action",action)
        """ EV """
        # 电动车到达时间9:30，离开时间13:30；空调开启时间11:45，关闭时间14：30
        # 电动车到达时间9:15，离开时间13:15；空调开启时间12:00，关闭时间15:00
        self.t_EV1_a, self.t_EV1_d= 37, 53
        # random
        # t_EV1_a1 = 34  # 8:15
        # t_EV1_a2 = 38  # 9:15
        # t_EV1_a = random.randint(t_EV1_a1, t_EV1_a2)
        # self.t_EV1_a = t_EV1_a
        # t_EV1_d1 = 46  # 11：15
        # t_EV1_d2 = 54  # 13：15
        # t_EV1_d = random.randint(t_EV1_d1, t_EV1_d2)
        # self.t_EV1_d = t_EV1_d

        energy_EV_init = 9.5
        energy_EV1 = self.EV.current_energy_EV(power_EV=power_EV1, t_EV_a=self.t_EV1_a, t_EV_d=self.t_EV1_d,
                                               energy_EV_init=energy_EV_init)
        if energy_EV1 == 0: power_EV1 = 0  # 电动车未达到充电时间，时间充电功率为0???
        self.power_EV = power_EV1
        self.energy_EV = energy_EV1
        """ AC """
        self.T_AC_min, self.T_AC_max = 23.1, 26.3
        self.t_AC1_e, self.t_AC1_l = 48, 60
        self.S_AC1 = self.AC.current_S_AC1(t_AC_e=self.t_AC1_e, t_AC_l=self.t_AC1_l, T_AC_min=self.T_AC_min, T_AC_max=self.T_AC_max)
        if self.S_AC1 == 0:
            # self.T_AC1 = self.temperature_out.current_temperature_out(self.day * self.iterations + self.time_step)
            power_AC1 = 0
            self.T_AC1 = self.AC.current_T_AC1(power_AC=power_AC1, a_AC=2.50, b_AC=17.7)
        else:
            self.T_AC1 = self.AC.current_T_AC1(power_AC=power_AC1, a_AC=2.50, b_AC=17.7)

        # self.T_AC1 = self.AC.current_T_AC1(power_AC=power_AC1, a_AC=2.50, b_AC=17.7)
        self.power_AC = power_AC1
        # if self.time_step==0: print("current_generation",self.generation.current_generation(self.day * self.iterations + self.time_step))
        # if self.time_step==0: print("current_temperature_out",self.temperature_out.current_temperature_out(self.day * self.iterations + self.time_step))
        """ grid_cost """
        if a_BES < 0:  # 放电
            power_discharge = abs(a_BES * DEFAULT_MAX_DISCHARGE)
            self.power_discharge = power_discharge  # 放电功率
            energy_discharge = self.power_discharge * self.interval
            energy_discharge = self.battery.supply(energy_discharge)  # 电池放电电量
            energy_discharge += (self.generation.current_generation(self.day * self.iterations + self.time_step) * 1)  # 光伏发电
            # energy_discharge += power_DG * self.interval  # CDG 发电
            energy_discharge -= (self.load.current_load(self.day * self.iterations + self.time_step) * 1)  # 负载耗电
            # =电池放电 + 光伏发电 - 负载耗电
            energy_discharge -= power_EV1 * self.interval  # 电动汽车
            energy_discharge -= power_AC1 * self.interval  # 空调
            if energy_discharge < 0:  # 购电
                cost_grid = self.grid.buy(energy_discharge)  # 负值
            elif energy_discharge > 0:  # 售电
                cost_grid = self.grid.sell(energy_discharge)  # 正值
            elif energy_discharge == 0:
                cost_grid = 0

        elif a_BES > 0:  # 充电
            power_charge = a_BES * self.generation.current_generation(self.day * self.iterations + self.time_step)
            self.power_charge = power_charge  # 充电功率
            #
            # energy_charge = self.power_charge * 1 + a_BES * 2 * 0.25
            energy_charge = self.power_charge * 1
            #
            energy_charge = self.battery.charge(energy_charge)  # 充电剩余电量
            energy_charge += ((1 - a_BES) * self.generation.current_generation(self.day * self.iterations + self.time_step) * 1)  # 光伏发电
            # energy_charge += power_DG * self.interval  # CDG 发电
            energy_charge -= (self.load.current_load(self.day * self.iterations + self.time_step) * 1)  # 负载耗电
            # =充电剩余电量 + 光伏发电 - 负载耗电
            energy_charge -= power_EV1 * self.interval  # 电动汽车
            energy_charge -= power_AC1 * self.interval  # 空调
            if energy_charge < 0:  # 购电
                cost_grid = self.grid.buy(energy_charge)
            elif energy_charge > 0:  # 售电
                cost_grid = self.grid.sell(energy_charge)
            elif energy_charge == 0:
                cost_grid = 0

        elif a_BES == 0:  # 购电
            energy_charge = (self.generation.current_generation(self.day * self.iterations + self.time_step)- self.load.current_load(self.day * self.iterations + self.time_step)) * 1
            # = 光伏发电 - 负载耗电
            energy_charge -= power_EV1 * self.interval  # 电动汽车
            energy_charge -= power_AC1 * self.interval  # 空调
            if energy_charge < 0:  # 购电
                cost_grid = self.grid.buy(energy_charge)
            elif energy_charge > 0:  # 售电
                cost_grid = self.grid.sell(energy_charge)
            elif energy_charge == 0:
                cost_grid = 0
        reward_1 = cost_grid

        #  reward_2
        SOC = self.battery.SoC
        if SOC <= 0.2 and SOC >= 0:
            reward_2 = math.exp(-abs(0.2 - SOC))
        elif SOC >= 0.9 and SOC <= 1:
            reward_2 = math.exp(-abs(0.9 - SOC))
        elif SOC > 0.2 and SOC < 0.9:
            reward_2 = 2 * math.exp(SOC)
        else:
            reward_2 = -100

        # reward_3
        lambda_EV1_1 = 1.12
        lambda_EV1_2 = 0.020

        if self.time_step + 1 == self.t_EV1_d:
            cost_EV = lambda_EV1_1 * self.energy_EV + lambda_EV1_2 * self.energy_EV * self.energy_EV
        elif self.time_step + 1 != self.t_EV1_d:
            cost_EV = 0
        reward_3 = - cost_EV
        # print("reward_3: ",reward_3)
        # reward_4
        lambda_AC_1,lambda_AC_2,T_AC_min,T_AC_max = 0.352, 0.396, 23.1, 26.3
        if T_AC_min <= self.T_AC1 <= T_AC_max:
            self.g_AC1=0
        elif self.T_AC1>T_AC_max:
            self.g_AC1=lambda_AC_1 * math.exp(lambda_AC_1*(self.T_AC1-T_AC_max))
        elif self.T_AC1<T_AC_min:
            self.g_AC1=lambda_AC_2 * math.exp(lambda_AC_2*(T_AC_min-self.T_AC1))
        if self.S_AC1==0:
            self.g_AC1 = 0
        elif self.S_AC1==1:
            self.g_AC1 = self.g_AC1
        reward_4 = - self.g_AC1
        # print("reward_4",reward_4)

        reward = reward_1 + reward_2 + 3*reward_3 + 3*reward_4
        self.r = reward
        self.r1 = reward_1
        self.r2 = reward_2
        self.r3 = reward_3*3
        self.r4 = reward_4*3
        # print("reward : ", reward)
        self.cost = cost_grid
        # Proceed to next timestep.
        self.time_step += 1
        # print("``````t``````",self.time_step)
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state()

        terminal = self.time_step == self.iterations

        info = self._build_info()
        # return state, reward / MAX_R, terminal, info
        return state, reward, terminal, info

    # def change_a(self,a):
    #     a = np.float(a)
    #     if self.generation.current_generation(self.day * self.iterations + self.time_step) == 0 and a > 0:
    #         a = 0
    #     return a

    def action_cost_soc(self):
        plot_action_BES = self.action_BES
        # if self.generation.current_generation(self.day * self.iterations + self.time_step) == 0 and plot_action_BES > 0:
        #     plot_action_BES = 0
        # if self.battery.SoC == 0 and plot_action_BES < 0:
        #     plot_action_BES = 0
        if plot_action_BES<0:
            plot_action_BES = plot_action_BES*0.25
        else:
            plot_action_BES = plot_action_BES*self.generation.current_generation(self.day * self.iterations + self.time_step)

        plot_SOC = float(self.battery.SoC)
        plot_cost = float(self.cost)
        # EV
        plot_power_EV = self.power_EV*0.25
        plot_energy_EV = self.energy_EV

        # AC
        plot_power_AC = self.power_AC*0.25
        Temperature_AC = self.T_AC1
        # print("cost_shape",np.shape(plot_cost))

        return plot_action_BES, plot_SOC, plot_cost, plot_power_EV, plot_energy_EV, plot_power_AC, Temperature_AC,self.time_step,self.t_EV1_a,self.t_EV1_d,self.T_AC_min, self.T_AC_max,self.t_AC1_e, self.t_AC1_l

    def record_loss(self):
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3
        r4 = self.r4
        r = self.r
        # g_AC1 = -self.g_AC1
        # return r1, r2, r3, r4, r, g_AC1
        return r1,r2,r3,r4,r

    def reset(self, total_steps,**kwargs):
        # print("main.total_steps",total_steps)
        self.total_steps = total_steps
        self.time_step = 0

        self.battery.reset()

        # self.AC.reset(self.time_step)

        ep = self.total_steps / 96
        global k
        k = int(ep%10) - 1
        if k<0: k=9
        # print("self.total_steps",self.total_steps)
        # print("k",k)
        global DEFAULT_OUTDOOR_TEMPERATURE
        global DEFAULT_POWER_GENERATED
        global DEFAULT_BASE_LOAD
        # 温度
        #DEFAULT_OUTDOOR_TEMPERATURE = np.genfromtxt("PV_MG1_G.csv", delimiter=',', skip_header=0, usecols=[1])
        # generation
        #DEFAULT_POWER_GENERATED = np.genfromtxt("PV_MG1_G.csv", delimiter=',', skip_header=0, usecols=[-1])
        # load
        #DEFAULT_BASE_LOAD = np.genfromtxt("load_MG1_G.csv", delimiter=',', skip_header=0, usecols=[-1])
        # print("DEFAULT_OUTDOOR_TEMPERATURE 1",DEFAULT_OUTDOOR_TEMPERATURE )

        DEFAULT_OUTDOOR_TEMPERATURE = DEFAULT_OUTDOOR_TEMPERATURE_old[96 * k:96 * k + 97]
        DEFAULT_POWER_GENERATED = DEFAULT_POWER_GENERATED_old[96 * k:96 * k + 97]
        DEFAULT_BASE_LOAD = DEFAULT_BASE_LOAD_old[96 * k:96 * k + 97]

        self.generation = Generation(kwargs.get("generation_data", DEFAULT_POWER_GENERATED))
        self.load = Load(kwargs.get("load_data", DEFAULT_BASE_LOAD))
        self.temperature_out = Temperature(kwargs.get("temperature_data", DEFAULT_OUTDOOR_TEMPERATURE))
        self.AC = AC(time_interval=self.interval, Temperature_out=self.temperature_out)
        self.AC.reset(self.time_step)
        # print("k",k)
        # if ep % 10 == 1:
        #     self.generation = self.generation[0:24]
        #     self.temperature_out = self.temperature_out[0:24]
        #     self.load = self.load[0:24]
        #     print("self.generation", self.generation)


        return self._build_state()

    def close(self):
        """
        Nothing to be done here, but has to be defined
        """
        return

    def seedy(self, s):
        """
        Set the random seed for consistent experiments
        """
        random.seed(s)
        np.random.seed(s)


if __name__ == '__main__':
    # Testing the environment
    # Initialize the environment
    env = MicroGridEnv()
    env.seedy(1)
    # Save the rewards in a list
    rewards = []
    # reset the environment to the initial state
    state = env.reset()
    plot_SOC = []
    while True:
        # Pick an action from the action space (here we pick an index between 0 and 80)
        # action = env.action_space.sample()
        # action =[np.argmax(action[0:4]),np.argmax(action[4:9]),np.argmax(action[9:11]),np.argmax(action[11:])]
        action = 0.3
        print("action", action)
        state, reward, terminal, _ = env.step(action)
        plot_SOC.append(state[2])
        print("reward", reward)
        rewards.append(reward)
        if terminal:
            break
    print("Total Reward:", sum(rewards))

    # Plot
    states = np.array(rewards)
    plt.plot(rewards)
    plt.title("rewards")
    plt.xlabel("Time")
    plt.ylabel("rewards")
    plt.show()

    # Plot
    plt.plot(plot_SOC)
    plt.title("soc")
    plt.xlabel("Time")
    plt.ylabel("soc")
    plt.show()
