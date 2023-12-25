from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import numpy as np
import torch
import gym
from ppo_chain_0 import PPO, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse

def str2bool(v):
    '''transfer str to bool for argparse''' # 为argparse模块转字符为布尔
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# class PLOT():
#     def __init__(self):
#         self.plot_action_BES = []
#         self.plot_money = []
#         self.money = 0
#         self.plot_soc = []
#         self.plot_power_EV = []
#         self.plot_energy_EV = []
#         self.plot_power_AC = []
#         self.plot_Temperature_AC = []
#         self.r1 = 0
#         self.r2 = 0
#         self.r3 = 0
#         self.r4 = 0
#         self.r = 0
#         self.g_AC1 = []
#     def record_MG(self):
#         plot_a_BES, plot_SOC, plot_C, plot_p_EV, plot_e_EV, plot_p_AC, Temperature_AC,self.time_step,self.t_EV1_a,self.t_EV1_d,self.T_AC_min, self.T_AC_max,self.t_AC1_e, self.t_AC1_l = env.action_cost_soc()
#         self.plot_action_BES.append(plot_a_BES)
#         self.plot_soc.append(plot_SOC)
#         self.money += plot_C
#         self.plot_money.append(self.money)
#         self.plot_power_EV.append(plot_p_EV)
#         self.plot_energy_EV.append(plot_e_EV)
#         self.plot_power_AC.append(plot_p_AC)
#         self.plot_Temperature_AC.append(Temperature_AC)
#         r1, r2, r3, r4, r, g_AC1 = env.record_loss()
#         self.r1 += r1
#         self.r2 += r2
#         self.r3 += r3
#         self.r4 += r4
#         self.r += r
#         self.g_AC1.append(g_AC1)
#     def plot_MG(self):
#         # 画图
#         """SCORE"""
#         # 设置图片大小
#         fig = plt.figure(figsize=(18.0, 6.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         plt.grid(True)
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(0, 2000)  # x轴范围设置
#         # plt.ylim(-3, 2)  # y轴范围设置
#         x_major_locator = MultipleLocator(100)  # x轴刻度线间隔
#         # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
#         # ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#
#         MG1_score_yuan = np.genfromtxt("MG1_score_yuan.csv", delimiter=',')  # 微网1单个运行
#         plt.plot(np.arange(len(MG1_score_yuan)), MG1_score_yuan, '-.', linewidth=2.0, alpha=1,
#                  label='Proposed')
#         # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
#         plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Episode', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         plt.ylabel('Episode reward', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#
#         """BES"""
#         # 设置图片大小
#         fig = plt.figure(figsize=(18.0, 6.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         # plt.grid(true)
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(0, 96)  # x轴范围设置
#         plt.ylim(-2, 2)  # y轴范围设置
#         x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
#         ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#
#         plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES,'-.',linewidth=2.0,alpha = 1, label = 'Action')
#         # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
#         plt.legend(loc='upper left',prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         plt.ylabel('Control Policy', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#         np.savetxt("plot_action_BES1.csv", self.plot_action_BES, delimiter=",", fmt="%.8f")
#
#
#         """SOC"""
#         # 设置图片大小
#         fig = plt.figure(figsize=(18.0, 6.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         # plt.grid(True)
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(0, 96)  # x轴范围设置
#         # plt.ylim(-3, 2)  # y轴范围设置
#         x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
#         # ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#
#         plt.plot(np.arange(len(self.plot_soc)), self.plot_soc, '-.', linewidth=2.0, alpha=1,
#                  label='Proposed')
#         # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
#         # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         plt.ylabel('SOC', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#         print("MG1_SOC:", self.plot_soc[-1])
#
#         """COST"""
#         # 设置图片大小
#         fig = plt.figure(figsize=(18.0, 6.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         # plt.grid(True)
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(0, 96)  # x轴范围设置
#         # plt.ylim(-3, 2)  # y轴范围设置
#         x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
#         # ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#
#         plt.plot(np.arange(len(self.plot_money)), self.plot_money, '-.', linewidth=2.0, alpha=1,
#                  label='Proposed')
#         # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
#         # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         plt.ylabel('Money(RMB)', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#         print("MG1_COST:", self.plot_money[-1])
#
#
#         """EV"""
#         # print("self.power_ev",self.plot_energy_EV)
#         # 设置图片大小
#         fig = plt.figure(figsize=(10.0, 7.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         plt.grid()
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(30, 60)  # x轴范围设置
#         plt.ylim(-1,15)  # y轴范围设置
#         x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         y_major_locator = MultipleLocator(5)  # 有y轴刻度线间隔
#         ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#
#         plt.plot(np.arange(len(self.plot_energy_EV)), self.plot_energy_EV, '-.', linewidth=2.0, alpha=1,
#                  label='Proposed')
#         list1 = [self.t_EV1_a-1]
#         list2 = [self.t_EV1_d-1]
#         plt.plot(list1,self.plot_energy_EV[self.t_EV1_a-1], label='Arrival', marker="o", markersize=15)  # marker设置标记形状 markersize设置标记大小
#         plt.plot(list2,self.plot_energy_EV[self.t_EV1_d-1], label='Departure', marker="o", markersize=15)
#         plt.text(self.t_EV1_a-1+0.5,self.plot_energy_EV[self.t_EV1_a-1]+0.2,'Arrival', fontdict={'family': 'Times New Roman', 'size': 20},verticalalignment='bottom', horizontalalignment='center')
#         plt.text(self.t_EV1_d-1+0.7,self.plot_energy_EV[self.t_EV1_d-1]+0.7,'Departure', fontdict={'family': 'Times New Roman', 'size': 20},verticalalignment='bottom', horizontalalignment='center')
#         # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
#         # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         plt.ylabel('Remaining charging demand/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#         print("MG1_energy_EV1:", self.plot_energy_EV[self.t_EV1_d-1])
#
#
#         """AC"""
#         # 设置图片大小
#         fig = plt.figure(figsize=(10.0, 7.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         # plt.grid(True)
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(0, 96)  # x轴范围设置
#         plt.ylim(0, 30)  # y轴范围设置
#         x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         y_major_locator = MultipleLocator(5)  # 有y轴刻度线间隔
#         ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#
#         plt.plot(np.arange(len(self.plot_Temperature_AC)), self.plot_Temperature_AC, linewidth=2.0, alpha=1,
#                  label='Proposed')
#         Temperature_OUT = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[1])
#         plt.plot(np.arange(len(Temperature_OUT)), Temperature_OUT, linewidth=2.0, alpha=1,
#                  label='T_out')
#         plt.axhline(self.T_AC_min, 0, 96,linewidth=1.0, alpha=1,label='T_min',color='black',linestyle='--')  # 横线
#         plt.axhline(self.T_AC_max, 0, 96,linewidth=1.0, alpha=1,label='T_max',color='black')  # 横线
#         plt.axvline(self.t_AC1_e, 0, 30, linewidth=1.0, alpha=0.2, color='cornflowerblue')  # 竖线
#         plt.axvline(self.t_AC1_l, 0, 30,linewidth=1.0, alpha=0.2,color='cornflowerblue')  # 竖线
#         # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
#         x = np.linspace(self.t_AC1_e,self.t_AC1_l)
#         plt.fill_between(x, 0, 30, facecolor='cornflowerblue', alpha=0.2)
#         list3 = [self.t_AC1_e]
#         list4 = [self.t_AC1_l]
#         plt.plot(list3, 30, marker=11,markersize=15)  # marker设置标记形状 markersize设置标记大小
#         plt.plot(list4, 30, marker=11, markersize=15)
#         plt.text(self.t_AC1_e, 31, 'Arrival',
#                  fontdict={'family': 'Times New Roman', 'size': 20}, verticalalignment='bottom',
#                  horizontalalignment='center')
#         plt.text(self.t_AC1_l, 31, 'Departure',
#                  fontdict={'family': 'Times New Roman', 'size': 20}, verticalalignment='bottom',
#                  horizontalalignment='center')
#         plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         plt.ylabel('Temperature', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#         np.savetxt("plot_Temperature_AC1.csv", self.plot_Temperature_AC, delimiter=",", fmt="%.8f")
#
#         """Power"""
#         # 设置图片大小
#         fig = plt.figure(figsize=(18.0, 6.0))
#         fig.patch.set_facecolor('white')  # 设置背景颜色
#         fig.patch.set_alpha(1)  # 设置透明度
#         bwith = 1  # 边框宽度设置为2
#         ax = plt.gca()  # 获取边框
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         # plt.grid(True)
#         # 同时设置
#         plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         plt.xlim(0, 96)  # x轴范围设置
#         plt.ylim(0, 5)  # y轴范围设置
#         x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
#         ax.yaxis.set_major_locator(y_major_locator)
#         ax.xaxis.set_major_locator(x_major_locator)
#         # 添加
#         plot_generated = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
#         plot_load = np.genfromtxt("load_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
#         # ax1 = fig.subplots()
#         # ax2 = ax1.twinx()
#         lin1_load=plt.plot(np.arange(len(plot_load)), plot_load, 'orange', label="Load1")
#         plt.fill_between(np.arange(len(plot_load)),
#                          plot_load,
#                          plot_load - plot_load,
#                          facecolor='orange',  # 填充颜色
#                          edgecolor='orange',  # 边界颜色
#                          alpha=0.2)
#         lin1_pv = plt.plot(np.arange(len(plot_generated)), plot_generated, 'green', label="PV1")
#         # ax1.fill_between(np.arange(len(plot_generated)),
#         #                  plot_generated,
#         #                  plot_generated - plot_generated,
#         #                  facecolor='green',  # 填充颜色
#         #                  edgecolor='green',  # 边界颜色
#         #                  alpha=0.3)
#         plot_power_EV = np.array(self.plot_power_EV)
#         lin1_ev=plt.plot(np.arange(len(self.plot_power_EV)), self.plot_power_EV, 'cornflowerblue', label="EV1")
#         plt.fill_between(np.arange(len(self.plot_power_EV)),
#                          self.plot_power_EV,
#                          plot_power_EV - plot_power_EV,
#                          facecolor='cornflowerblue',  # 填充颜色
#                          edgecolor='cornflowerblue',  # 边界颜色
#                          alpha=0.3)
#         plot_power_AC = np.array(self.plot_power_AC)
#         lin1_ac=plt.plot(np.arange(len(self.plot_power_AC)), self.plot_power_AC, 'red', label="AC1")
#         plt.fill_between(np.arange(len(self.plot_power_AC)),
#                          self.plot_power_AC,
#                          plot_power_AC - plot_power_AC,
#                          facecolor='red',  # 填充颜色
#                          edgecolor='red',  # 边界颜色
#                          alpha=0.2)
#
#         ax2 = ax.twinx()
#         # plt.ylim(-2, 2)  # y轴范围设置
#         # 设置边框
#         ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
#         ax.spines['left'].set_linewidth(bwith)  # 图框左边
#         ax.spines['top'].set_linewidth(bwith)  # 图框上边
#         ax.spines['right'].set_linewidth(bwith)  # 图框右边
#         # 显示网格线
#         # plt.grid(True)
#         # 同时设置
#         # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # 单独设置x和y
#         # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#         # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
#
#         plt.xticks(fontproperties='Times New Roman', size=18)
#         plt.yticks(fontproperties='Times New Roman', size=18)
#         # 刻度线的大小长短粗细
#         plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
#         # 不显示刻度标签
#         # ax.axes.xaxis.set_ticklabels([])
#         # ax.axes.yaxis.set_ticklabels([])
#         # plt.xlim(0, 95)  # x轴范围设置
#         plt.ylim(-2, 2)  # y轴范围设置
#         # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
#         y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
#         ax2.yaxis.set_major_locator(y_major_locator)
#         # ax.xaxis.set_major_locator(x_major_locator)
#         lin2=ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
#                  label='BES1')
#         ax2.set_ylabel("BES policy", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         ax.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         # 添加
#
#         lines = lin1_pv+lin1_load+lin1_ac+lin1_ev + lin2
#         labs = [label.get_label() for label in lines]
#         ax.legend(lines,labs,loc='best', prop={'family': 'Times New Roman', 'size': 20})
#         # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
#         plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
#         # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
#         plt.show()
#
#         print("r1", self.r1)
#         print("r2", self.r2)
#         print("r3", self.r3)
#         print("r4", self.r4)
#         print("r", self.r)
#         # print("g_AC1", self.g_AC1)
DEFAULT_OUTDOOR_TEMPERATURE_old = np.genfromtxt("PV_MG1_G.csv", delimiter=',',skip_header=0, usecols=[1])
# generation
DEFAULT_POWER_GENERATED_old = np.genfromtxt("PV_MG1_G.csv", delimiter=',', skip_header=0, usecols=[-1])
# load
DEFAULT_BASE_LOAD_old = np.genfromtxt("load_MG1_G.csv", delimiter=',', skip_header=0, usecols=[-1])
# print("DEFAULT_OUTDOOR_TEMPERATURE 1",DEFAULT_OUTDOOR_TEMPERATURE )
# global k
k = 8
# DEFAULT_OUTDOOR_TEMPERATURE=DEFAULT_OUTDOOR_TEMPERATURE[0:97]
# DEFAULT_POWER_GENERATED=DEFAULT_POWER_GENERATED[0:97]
# DEFAULT_BASE_LOAD = DEFAULT_BASE_LOAD[0:97]
DEFAULT_OUTDOOR_TEMPERATUREm1=DEFAULT_OUTDOOR_TEMPERATURE_old[96*k:96*k+97]
DEFAULT_POWER_GENERATEDm1=DEFAULT_POWER_GENERATED_old[96*k:96*k+97]
DEFAULT_BASE_LOADm1 = DEFAULT_BASE_LOAD_old[96*k:96*k+97]
# print("DEFAULT_OUTDOOR_TEMPERATURE",DEFAULT_OUTDOOR_TEMPERATUREm)
class PLOT():
    def __init__(self):
        self.plot_action_BES = []
        self.plot_money = []
        self.money = 0
        self.plot_soc = []
        self.plot_power_EV = []
        self.plot_energy_EV = []
        self.plot_power_AC = []
        self.plot_Temperature_AC = []
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0
        self.r4 = 0
        self.r = 0
    def record_MG(self):
        plot_a_BES, plot_SOC, plot_C, plot_p_EV, plot_e_EV, plot_p_AC, Temperature_AC,self.time_step,self.t_EV1_a,self.t_EV1_d,self.T_AC_min, self.T_AC_max,self.t_AC1_e, self.t_AC1_l = env.action_cost_soc()
        self.plot_action_BES.append(plot_a_BES)
        self.plot_soc.append(plot_SOC)
        self.money += plot_C
        self.plot_money.append(self.money)
        self.plot_power_EV.append(plot_p_EV)
        self.plot_energy_EV.append(plot_e_EV)
        self.plot_power_AC.append(plot_p_AC)
        self.plot_Temperature_AC.append(Temperature_AC)
        r1, r2, r3, r4, r = env.record_loss()
        self.r1 += r1
        self.r2 += r2
        self.r3 += r3
        self.r4 += r4
        self.r += r
    def plot_MG(self):
        # # 画图
        """SCORE"""
        # 设置图片大小
        fig = plt.figure(figsize=(18.0, 6.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        plt.grid(True)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 2000)  # x轴范围设置
        # plt.ylim(-3, 2)  # y轴范围设置
        x_major_locator = MultipleLocator(100)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        MG1_score_yuan = np.genfromtxt("MG1_score_yuan.csv", delimiter=',')  # 微网1单个运行
        plt.plot(np.arange(len(MG1_score_yuan)), MG1_score_yuan, '-.', linewidth=2.0, alpha=1,
                 label='Proposed')
        # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
        plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Episode', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        plt.ylabel('Episode reward', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()

        """BES"""
        # 设置图片大小
        fig = plt.figure(figsize=(18.0, 6.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(true)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 95)  # x轴范围设置
        plt.ylim(-2, 2)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES,'-.',linewidth=2.0,alpha = 1, label = 'Action')
        # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
        plt.legend(loc='upper left',prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        plt.ylabel('Control Policy', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()
        np.savetxt("plot_action_BES1.csv", self.plot_action_BES, delimiter=",", fmt="%.8f")


        """SOC"""
        # 设置图片大小
        fig = plt.figure(figsize=(18.0, 6.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 95)  # x轴范围设置
        # plt.ylim(-3, 2)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        plt.plot(np.arange(len(self.plot_soc)), self.plot_soc, '-.', linewidth=2.0, alpha=1,
                 label='Proposed')
        # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
        # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        plt.ylabel('SOC', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()
        print("MG1_SOC:", self.plot_soc[-1])

        """COST"""
        # 设置图片大小
        fig = plt.figure(figsize=(18.0, 6.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 95)  # x轴范围设置
        # plt.ylim(-3, 2)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        plt.plot(np.arange(len(self.plot_money)), self.plot_money, '-.', linewidth=2.0, alpha=1,
                 label='Proposed')
        # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
        # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        plt.ylabel('Money(RMB)', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()
        print("MG1_COST:", self.plot_money[-1])


        """EV"""
        # print("self.power_ev",self.plot_energy_EV)
        # 设置图片大小
        fig = plt.figure(figsize=(10.0, 7.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        plt.grid()
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])

        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(30, 60)  # x轴范围设置
        plt.ylim(-1,15)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(5)  # 有y轴刻度线间隔
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        plt.plot(np.arange(len(self.plot_energy_EV)), self.plot_energy_EV, '-.', linewidth=2.0, alpha=1,
                 label='Proposed')
        list1 = [self.t_EV1_a-1]
        list2 = [self.t_EV1_d-1]
        plt.plot(list1,self.plot_energy_EV[self.t_EV1_a-1], label='Arrival', marker="o", markersize=15)  # marker设置标记形状 markersize设置标记大小
        plt.plot(list2,self.plot_energy_EV[self.t_EV1_d-1], label='Departure', marker="o", markersize=15)
        plt.text(self.t_EV1_a-1+0.5,self.plot_energy_EV[self.t_EV1_a-1]+0.2,'Arrival', fontdict={'family': 'Times New Roman', 'size': 20},verticalalignment='bottom', horizontalalignment='center')
        plt.text(self.t_EV1_d-1+0.7,self.plot_energy_EV[self.t_EV1_d-1]+0.7,'Departure', fontdict={'family': 'Times New Roman', 'size': 20},verticalalignment='bottom', horizontalalignment='center')
        # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
        # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        plt.ylabel('Remaining charging demand/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()
        print("MG1_energy_EV1:", self.plot_energy_EV[self.t_EV1_d-1])


        """AC"""
        # 设置图片大小
        fig = plt.figure(figsize=(10.0, 7.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 95)  # x轴范围设置
        plt.ylim(0, 30)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(5)  # 有y轴刻度线间隔
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        plt.plot(np.arange(len(self.plot_Temperature_AC)), self.plot_Temperature_AC, linewidth=2.0, alpha=1,
                 label='Proposed')
        # Temperature_OUT = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[1])
        Temperature_OUT = DEFAULT_OUTDOOR_TEMPERATURE
        # plt.plot(np.arange(len(Temperature_OUT)), Temperature_OUT, linewidth=2.0, alpha=1,
        #          label='T_out')
        plt.plot(np.arange(len(DEFAULT_OUTDOOR_TEMPERATUREm1)), DEFAULT_OUTDOOR_TEMPERATUREm1, linewidth=2.0, alpha=1,
                 label='T_out')
        # print("Temperature_OUT",DEFAULT_OUTDOOR_TEMPERATUREm)
        plt.axhline(self.T_AC_min, 0, 95,linewidth=1.0, alpha=1,label='T_min',color='black',linestyle='--')  # 横线
        plt.axhline(self.T_AC_max, 0, 95,linewidth=1.0, alpha=1,label='T_max',color='black')  # 横线
        plt.axvline(self.t_AC1_e, 0, 30, linewidth=1.0, alpha=0.2, color='cornflowerblue')  # 竖线
        plt.axvline(self.t_AC1_l, 0, 30,linewidth=1.0, alpha=0.2,color='cornflowerblue')  # 竖线
        # color是填充色，edgecolors是边框颜色，marker是标记，s是大小，alpha是透明度，label是图例要显示的内容
        x = np.linspace(self.t_AC1_e,self.t_AC1_l)
        plt.fill_between(x, 0, 30, facecolor='cornflowerblue', alpha=0.2)
        list3 = [self.t_AC1_e]
        list4 = [self.t_AC1_l]
        plt.plot(list3, 30, marker=11,markersize=15)  # marker设置标记形状 markersize设置标记大小
        plt.plot(list4, 30, marker=11, markersize=15)
        plt.text(self.t_AC1_e, 31, 'Arrival',
                 fontdict={'family': 'Times New Roman', 'size': 20}, verticalalignment='bottom',
                 horizontalalignment='center')
        plt.text(self.t_AC1_l, 31, 'Departure',
                 fontdict={'family': 'Times New Roman', 'size': 20}, verticalalignment='bottom',
                 horizontalalignment='center')
        plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        plt.ylabel('Temperature', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()
        np.savetxt("plot_Temperature_AC1.csv", self.plot_Temperature_AC, delimiter=",", fmt="%.8f")

        # """Power未叠加"""
        # # 设置图片大小
        # fig = plt.figure(figsize=(18.0, 6.0))
        # fig.patch.set_facecolor('white')  # 设置背景颜色
        # fig.patch.set_alpha(1)  # 设置透明度
        # bwith = 1  # 边框宽度设置为2
        # ax = plt.gca()  # 获取边框
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(-1, 7)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # # 添加
        # plot_generated1 = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        # plot_generated = []
        # for i_plot_generated in plot_generated1:
        #     i_plot_generated = -i_plot_generated
        #     plot_generated.append(i_plot_generated)
        # plot_load = np.genfromtxt("load_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        # # ax1 = fig.subplots()
        # # ax2 = ax1.twinx()
        # lin1_load = plt.plot(np.arange(len(plot_load)), plot_load, 'orange', label="Load")
        # plt.fill_between(np.arange(len(plot_load)),
        #                  plot_load,
        #                  plot_load - plot_load,
        #                  facecolor='orange',  # 填充颜色
        #                  edgecolor='orange',  # 边界颜色
        #                  alpha=0.2)
        # lin1_pv = plt.plot(np.arange(len(plot_generated)), plot_generated, 'green', label="PV")
        # plt.fill_between(np.arange(len(plot_generated)),
        #                  plot_generated,
        #                  plot_load - plot_load,
        #                  facecolor='green',  # 填充颜色
        #                  edgecolor='green',  # 边界颜色
        #                  alpha=0.3)
        # plot_power_EV = np.array(self.plot_power_EV)
        # lin1_ev = plt.plot(np.arange(len(self.plot_power_EV)), self.plot_power_EV, 'cornflowerblue', label="EV")
        # plt.fill_between(np.arange(len(self.plot_power_EV)),
        #                  self.plot_power_EV,
        #                  plot_power_EV - plot_power_EV,
        #                  facecolor='cornflowerblue',  # 填充颜色
        #                  edgecolor='cornflowerblue',  # 边界颜色
        #                  alpha=0.3)
        # plot_power_AC = np.array(self.plot_power_AC)
        # lin1_ac = plt.plot(np.arange(len(self.plot_power_AC)), self.plot_power_AC, 'red', label="AC1")
        # plt.fill_between(np.arange(len(self.plot_power_AC)),
        #                  self.plot_power_AC,
        #                  plot_power_AC - plot_power_AC,
        #                  facecolor='red',  # 填充颜色
        #                  edgecolor='red',  # 边界颜色
        #                  alpha=0.2)
        # # lin2 = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
        # #                 label='BES1')
        # plot_power_EV = np.array(self.plot_action_BES)
        # lin1_bes = plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, 'grey', label="BES")
        # plt.fill_between(np.arange(len(self.plot_action_BES)),
        #                  self.plot_action_BES,
        #                  plot_power_EV - plot_power_EV,
        #                  facecolor='grey',  # 填充颜色
        #                  edgecolor='grey',  # 边界颜色
        #                  alpha=0.2)
        #
        # ax2 = ax.twinx()
        # # plt.ylim(-2, 2)  # y轴范围设置
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(0, 1)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(0.2)  # 有y轴刻度线间隔
        # ax2.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # buy_price = np.genfromtxt("buy_price.csv", delimiter=',', skip_header=1,usecols=[-1])  # 微网1单个运行
        # # plt.plot(np.arange(len(buy_price)), buy_price, linewidth=2.0, alpha=1, color='#1f77b4',
        # #          label='buy_price')
        # lin2 = ax2.step(np.arange(len(buy_price)), buy_price, '-.', linewidth=2.0, alpha=1, color='#1f77b4',
        #                 label='Price')
        # ax2.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # # 添加
        #
        # lines = lin1_pv + lin1_load + lin1_ac + lin1_ev + lin1_bes
        # labs = [label.get_label() for label in lines]
        # ax.legend(lines, labs, loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        # plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        # # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # plt.show()


        # #power2 三个坐标
        # fig = plt.figure(figsize=(18.0, 6.0))
        # ax01 = HostAxes(fig, [0.1, 0.1, 0.7, 0.7])  # 用[left, bottom, weight, height]的方式定义axes, 0 <= l,b,w,h <= 1
        # kw = dict(linewidth=2, markerfacecolor='none', markersize=4)
        #
        # # parasite addtional axes, share x
        # ax02 = ParasiteAxes(ax01, sharex=ax01)
        # ax03 = ParasiteAxes(ax01, sharex=ax01)
        #
        # # append axes
        # ax01.parasites.append(ax02)
        # ax01.parasites.append(ax03)
        #
        # ax03_axisline = ax02.get_grid_helper().new_fixed_axis
        # ax03.axis['right4'] = ax03_axisline(loc='right', axes=ax03, offset=(60, 0))
        #
        # fig.add_axes(ax01)
        # #
        # # 设置图片大小
        # # fig = plt.figure(figsize=(18.0, 6.0))
        # # fig.patch.set_facecolor('white')  # 设置背景颜色
        # # fig.patch.set_alpha(1)  # 设置透明度
        # bwith = 1  # 边框宽度设置为2
        # ax = plt.gca()  # 获取边框
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # ax03.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # # ax01.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax01.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax02.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax02.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax03.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax03.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # # plt.xticks(fontproperties='Times New Roman', size=18)
        # # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # ax01.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # ax02.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # ax03.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # set xlim for yaxis
        # # ax01.set_xlim(0, 96)
        # # ax01.set_ylim(0, 5)
        # # ax02.set_ylim(-2, 5)
        # # ax03.set_ylim(-2, 1)
        # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # ax01.set_xlim(0,96)
        # # plt.ylim(0, 5)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # # ax.yaxis.set_major_locator(y_major_locator)
        # ax01.xaxis.set_major_locator(x_major_locator)
        #
        # ax01.axes.yaxis.set_ticklabels([])
        # ax02.axes.yaxis.set_ticklabels([])
        # ax03.axes.yaxis.set_ticklabels([])
        # ax01.set_ylim(0, 5)
        # ax02.set_ylim(-2, 5)
        # ax03.set_ylim(-2, 1)
        # # plt.ylim(0, 5)  # y轴范围设置
        # # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # # ax.yaxis.set_major_locator(y_major_locator)
        # # ax01.xaxis.set_major_locator(x_major_locator)
        # #
        # plot_generated = []
        # # for i_plot_generated in plot_generated1:
        # #     i_plot_generated = -i_plot_generated
        # #     plot_generated.append(i_plot_generated)
        # h1, = ax01.plot(np.arange(len(plot_generated1)), plot_generated1, '-.', linewidth=2.0,alpha=0.6, color='green', label="PV")
        # plt.fill_between(np.arange(len(plot_generated1)),
        #                  plot_generated1,
        #                  plot_generated1 - plot_generated1,
        #                  facecolor='green',  # 填充颜色
        #                  edgecolor='green',  # 边界颜色
        #                  alpha=0.6)
        # h11, = ax01.plot(np.arange(len(plot_load)), plot_load,linewidth=2.0,alpha=0.8, color='green', label="Load")
        # plt.fill_between(np.arange(len(plot_load)),
        #                  plot_load,
        #                  plot_load - plot_load,
        #                  facecolor='green',  # 填充颜色
        #                  edgecolor='green',  # 边界颜色
        #                  alpha=0.8)
        # h2, = ax02.plot(np.arange(len(self.plot_power_EV)), self.plot_power_EV, linewidth=2, alpha=0.6,color='orange', label="EV")
        # ax02.fill_between(np.arange(len(self.plot_power_EV)),
        #                  self.plot_power_EV,
        #                  plot_power_EV - plot_power_EV,
        #                  facecolor='orange',  # 填充颜色
        #                  edgecolor='orange',  # 边界颜色
        #                  alpha=0.6)
        # h22, = ax02.plot(np.arange(len(self.plot_power_AC)), self.plot_power_AC, linewidth=2, alpha=0.8,color='orange',
        #                 label="AC")
        # ax02.fill_between(np.arange(len(self.plot_power_AC)),
        #                  self.plot_power_AC,
        #                  plot_power_AC - plot_power_AC,
        #                  facecolor='orange',  # 填充颜色
        #                  edgecolor='orange',  # 边界颜色
        #                  alpha=0.8)
        # h222, = ax02.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, linewidth=2.0, alpha=1, color='orange',
        #                  label="BES1")
        # # buy_price1=[]
        # # for i in range(0,96):
        # #     buy_price1.append(buy_price[i])
        # h3 = ax03.step(np.arange(len(buy_price)), buy_price, linewidth=2, alpha=1,color='cornflowerblue', label='Price')
        # # h3 = ax03.plot(np.arange(len(buy_price)), buy_price, linewidth=2, color='black', label='Price')
        #
        # # invisible right axis of ax
        # ax.axis['right'].set_visible(False)
        # ax.axis['top'].set_visible(True)
        # ax.axis['right'].set_visible(True)
        # ax.axis['right'].major_ticklabels.set_visible(True)
        #
        # # set label for axis
        # ax01.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax01.set_xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax02.set_ylabel("Control policy", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax03.set_ylabel('Price', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        #
        # # set xlim for yaxis
        # # ax01.set_xlim(0, 96)
        # # ax01.set_ylim(0, 5)
        # # ax02.set_ylim(-2, 5)
        # # ax03.set_ylim(-2, 1)
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # # plt.xlim(0, 96)  # x轴范围设置
        # ax01.set_xlim(0, 96)
        # # plt.ylim(0, 5)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # # ax.yaxis.set_major_locator(y_major_locator)
        # ax01.xaxis.set_major_locator(x_major_locator)
        #
        # ax01.axes.yaxis.set_ticklabels([])
        # ax02.axes.yaxis.set_ticklabels([])
        # ax03.axes.yaxis.set_ticklabels([])
        # ax01.set_ylim(0, 5)
        # ax02.set_ylim(-2, 5)
        # ax03.set_ylim(-2, 1)
        #
        # ax01.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)

        # # name axies, xticks colors
        # ax01.axis['left'].label.set_color('green')
        # ax02.axis['right'].label.set_color('orange')
        # ax03.axis['right4'].label.set_color('cornflowerblue')
        #
        # ax01.axis['left'].major_ticks.set_color('green')
        # ax02.axis['right'].major_ticks.set_color('orange')
        # ax03.axis['right4'].major_ticks.set_color('cornflowerblue')
        #
        # ax01.axis['left'].major_ticklabels.set_color('green')
        # ax02.axis['right'].major_ticklabels.set_color('orange')
        # ax03.axis['right4'].major_ticklabels.set_color('cornflowerblue')
        #
        # ax01.axis['left'].line.set_color('green')
        # ax02.axis['right'].line.set_color('orange')
        # ax03.axis['right4'].line.set_color('cornflowerblue')
        # plt.show()

        # """Power"""
        # # 设置图片大小
        # # fig = plt.figure(figsize=(18.0, 6.0))
        # fig = plt.figure(figsize=(10.0, 7.0))
        # fig.patch.set_facecolor('white')  # 设置背景颜色
        # fig.patch.set_alpha(1)  # 设置透明度
        # bwith = 1  # 边框宽度设置为2
        # ax = plt.gca()  # 获取边框
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(-2, 5)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # # 添加
        # plot_generated1 = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        # plot_generated = []
        # for i_plot_generated in plot_generated1:
        #     i_plot_generated = -i_plot_generated
        #     plot_generated.append(i_plot_generated)
        # plot_load = np.genfromtxt("load_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        # # ax1 = fig.subplots()
        # # ax2 = ax1.twinx()
        # lin1_load=plt.plot(np.arange(len(plot_load)), plot_load, 'orange', label="Load1")
        # plt.fill_between(np.arange(len(plot_load)),
        #                  plot_load,
        #                  plot_load - plot_load,
        #                  facecolor='orange',  # 填充颜色
        #                  edgecolor='orange',  # 边界颜色
        #                  alpha=0.2)
        # lin1_pv = plt.plot(np.arange(len(plot_generated)), plot_generated, 'green', label="PV1")
        # plt.fill_between(np.arange(len(plot_generated)),
        #                  plot_generated,
        #                  plot_load - plot_load,
        #                  facecolor='green',  # 填充颜色
        #                  edgecolor='green',  # 边界颜色
        #                  alpha=0.3)
        # plot_power_EV = np.array(self.plot_power_EV)
        # self.plot_power_EV1 = []
        # self.plot_load1 = []
        # self.plot_power_AC1 = []
        # for i in range(0,96):
        #     i_plot_power_EV = self.plot_power_EV[i]
        #     i_plot_load = plot_load[i]
        #     i_plot_power_AC = self.plot_power_AC[i]
        #     i_plot_power_EV = i_plot_power_EV +i_plot_load
        #     i_plot_power_AC = i_plot_power_AC +i_plot_power_EV
        #     self.plot_load1.append(i_plot_load)
        #     self.plot_power_EV1.append(i_plot_power_EV)
        #     self.plot_power_AC1.append(i_plot_power_AC)
        #
        # lin1_ev=plt.plot(np.arange(len(self.plot_power_EV1)), self.plot_power_EV1, 'cornflowerblue', label="EV1")
        # plt.fill_between(np.arange(len(self.plot_power_EV1)),
        #                  self.plot_power_EV1,
        #                  self.plot_load1,
        #                  facecolor='cornflowerblue',  # 填充颜色
        #                  edgecolor='cornflowerblue',  # 边界颜色
        #                  alpha=0.3)
        #
        # plot_power_AC = np.array(self.plot_power_AC1)
        # lin1_ac=plt.plot(np.arange(len(self.plot_power_AC1)), self.plot_power_AC1, 'red', label="AC1")
        # plt.fill_between(np.arange(len(self.plot_power_AC1)),
        #                  self.plot_power_AC1,
        #                  self.plot_power_EV1,
        #                  facecolor='red',  # 填充颜色
        #                  edgecolor='red',  # 边界颜色
        #                  alpha=0.2)
        # # lin2 = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
        # #                 label='BES1')
        # plot_power_EV = np.array(self.plot_action_BES)
        # lin1_bes = plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, 'grey', label="BES")
        # # plt.fill_between(np.arange(len(self.plot_action_BES)),
        # #                  self.plot_action_BES,
        # #                  plot_power_EV - plot_power_EV,
        # #                  facecolor='grey',  # 填充颜色
        # #                  edgecolor='grey',  # 边界颜色
        # #                  alpha=0.2)
        #
        # ax2 = ax.twinx()
        # # plt.ylim(-2, 2)  # y轴范围设置
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(0, 1)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(0.2)  # 有y轴刻度线间隔
        # ax2.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # buy_price = np.genfromtxt("buy_price.csv", delimiter=',',skip_header=1)  # 微网1单个运行
        # # plt.plot(np.arange(len(buy_price)), buy_price, linewidth=2.0, alpha=1, color='#1f77b4',
        # #          label='buy_price')
        # lin2=ax2.step(np.arange(len(buy_price)), buy_price, '-.', linewidth=2.0, alpha=1,color='#1f77b4',
        #          label='Price')
        # ax2.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # # 添加
        #
        # lines = lin1_pv+lin1_load+lin1_ac+lin1_ev + lin1_bes
        # labs = [label.get_label() for label in lines]
        # ax.legend(lines,labs,loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        # plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        # # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # plt.show()

        # power3
        # # 构造多个ax
        # # fig,ax1 = plt.subplots(figsize = cm2inch(16,9))
        # fig, ax1 = plt.subplots(figsize=(45 / 2.54, 15 / 2.54))
        # fig.patch.set_facecolor('white')  # 设置背景颜色
        # fig.patch.set_alpha(1)  # 设置透明度
        # bwith = 1  # 边框宽度设置为2
        # ax2 = ax1.twinx()
        # ax3 = ax1.twinx()
        # # ax4 = ax1.twinx()
        # ax = plt.gca()  # 获取边框
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # ax1.spines['right'].set_linewidth(bwith)  # 图框右边
        # ax2.spines['right'].set_linewidth(bwith)  # 图框右边
        # ax3.spines['right'].set_linewidth(bwith)  # 图框右边
        #
        # # 将构造的ax右侧的spine向右偏移
        # ax3.spines['right'].set_position(('outward', 60))
        # # ax4.spines['right'].set_position(('outward',120))
        #
        # # 绘制
        # # img1, = ax1.plot(x, y1, c='tab:blue')
        # # img2, = ax2.plot(x, y2, c='tab:orange')
        # # img3, = ax3.plot(x, y3, c='tab:green')
        # img1, = ax1.plot(np.arange(len(plot_generated1)), plot_generated1, c='tab:blue',linewidth=2.0, alpha=0.6, label="PV")
        # ax1.fill_between(np.arange(len(plot_generated1)),
        #                  plot_generated1,
        #                  plot_generated1 - plot_generated1,
        #                  facecolor='blue',  # 填充颜色
        #                  edgecolor='blue',  # 边界颜色
        #                  alpha=0.6)
        # img2, = ax1.plot(np.arange(len(plot_load)), plot_load, c='tab:blue', linewidth=2.0, alpha=0.8, label="Load")
        # ax1.fill_between(np.arange(len(plot_load)),
        #                  plot_load,
        #                  plot_load - plot_load,
        #                  facecolor='blue',  # 填充颜色
        #                  edgecolor='blue',  # 边界颜色
        #                  alpha=0.8)
        #
        # img3, = ax2.plot(np.arange(len(self.plot_power_EV)), self.plot_power_EV, c='tab:orange',linewidth=2, alpha=0.6,
        #                 label="EV")
        # ax2.fill_between(np.arange(len(self.plot_power_EV)),
        #                   self.plot_power_EV,
        #                   plot_power_EV - plot_power_EV,
        #                   facecolor='orange',  # 填充颜色
        #                   edgecolor='orange',  # 边界颜色
        #                   alpha=0.6)
        # img4, = ax2.plot(np.arange(len(self.plot_power_AC)), self.plot_power_AC, c='tab:orange',linewidth=2, alpha=0.8,
        #                  label="AC")
        # ax2.fill_between(np.arange(len(self.plot_power_AC)),
        #                   self.plot_power_AC,
        #                   plot_power_AC - plot_power_AC,
        #                   facecolor='orange',  # 填充颜色
        #                   edgecolor='orange',  # 边界颜色
        #                   alpha=0.8)
        # img5, = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES,c='tab:orange', linewidth=2.0, alpha=1,
        #                   label="BES1")
        # buy_price1=[]
        # for i in range(0,96):
        #     buy_price1.append(buy_price[i])
        # img6, = ax3.step(np.arange(len(buy_price)), buy_price, c='tab:green',linewidth=2, alpha=1,
        #                label='Price')
        # # h3 = ax03.plot(np.arange(len(buy_price)), buy_price, linewidth=2, color='black', label='Price')
        #
        # # img4, = ax4.plot(x,y4,c = 'tab:red')
        #
        # # 获取对应折线图颜色给到spine ylabel yticks yticklabels
        # axs = [ax1, ax2, ax3]
        # imgs = [img1,img2,img3,img4,img5,img6]
        # for i in range(len(axs)):
        #     if i == 0: j=0
        #     if i == 1: j = 2
        #     if i == 2: j = 5
        #     axs[i].spines['right'].set_color(imgs[j].get_color())
        #     # axs[i].set_ylabel('y{}'.format(i + 1), c=imgs[i].get_color(),
        #     #                   fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        #     axs[i].tick_params(axis='y', color=imgs[j].get_color(), labelcolor=imgs[j].get_color())
        #     axs[i].spines['left'].set_color(img1.get_color())  # 注意ax1是left
        # # 设置横纵坐标的名称以及对应字体格式
        # labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels() + ax3.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]
        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 18,
        #          }
        # # plt.xlabel('round', font2)
        #
        # # 刻度线的大小长短粗细
        # ax1.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # ax2.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # ax3.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # ax1.set_xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax1.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax2.set_ylabel("Control Policy", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax3.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax1.set_ylim(0, 5)
        # ax1.set_xlim(0, 96)
        # ax2.set_ylim(-2, 7)
        # ax3.set_ylim(0, 1)
        # # ax4.set_ylim(0,10000)
        # plt.tight_layout()
        # # ax1.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # ax2.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # ax3.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # lines = img1 + img2 + img3 + img4 + img5 +img6
        # # labs = [label.get_label() for label in lines]
        # # ax.legend(lines, labs, loc='best', prop={'family': 'Times New Roman', 'size': 20})
        #
        #
        # # plt.savefig('n axis.png',dpi = 600)
        # plt.show()
        """Power叠加"""
        # # 设置图片大小
        # # fig = plt.figure(figsize=(10.0, 7.0))
        # fig = plt.figure(figsize=(18.0, 6.0))
        # fig.patch.set_facecolor('white')  # 设置背景颜色
        # fig.patch.set_alpha(1)  # 设置透明度
        # bwith = 1  # 边框宽度设置为2
        # ax = plt.gca()  # 获取边框
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(-2, 6.8)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # # 添加
        # plot_generated1 = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        # for i in range(0, 96):
        #     plot_generated1[i] = DEFAULT_POWER_GENERATEDm1[i]
        # plot_generated = []
        # standard0 = []
        # for i in range(0, 96):
        # # for i_plot_generated in plot_generated1:
        #     i_plot_generated = -plot_generated1[i]
        #     plot_generated.append(i_plot_generated)
        #     standard0.append(0)
        # plot_load1 = np.genfromtxt("load_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        # for i in range(0, 96):
        #     plot_load1[i] = DEFAULT_BASE_LOADm1[i]
        # plot_load=[]
        # for i in range(0, 96):
        # # for i_plot_generated in plot_generated1:
        #     i_plot_load = plot_load1[i]
        #     plot_load.append(i_plot_load)
        # # ax1 = fig.subplots()
        # # ax2 = ax1.twinx()
        # lin1_load = plt.plot(np.arange(len(plot_load)), plot_load, 'orange', label="Load")
        # plt.fill_between(np.arange(len(plot_load)),
        #                  plot_load,
        #                  # plot_load - plot_load,
        #                  standard0,
        #                  facecolor='orange',  # 填充颜色
        #                  edgecolor='orange',  # 边界颜色
        #                  alpha=0.2)
        # lin1_pv = plt.plot(np.arange(len(plot_generated)), plot_generated, 'green', label="PV")
        # plt.fill_between(np.arange(len(plot_generated)),
        #                  plot_generated,
        #                  # plot_load - plot_load,
        #                  standard0,
        #                  facecolor='green',  # 填充颜色
        #                  edgecolor='green',  # 边界颜色
        #                  alpha=0.3)
        # plot_power_EV = np.array(self.plot_power_EV)
        # self.plot_power_EV1 = []
        # self.plot_load1 = []
        # self.plot_power_AC1 = []
        # for i in range(0, 96):
        #     i_plot_power_EV = self.plot_power_EV[i]
        #     i_plot_load = plot_load[i]
        #     i_plot_power_AC = self.plot_power_AC[i]
        #     i_plot_power_EV = i_plot_power_EV + i_plot_load
        #     i_plot_power_AC = i_plot_power_AC + i_plot_power_EV
        #     self.plot_load1.append(i_plot_load)
        #     self.plot_power_EV1.append(i_plot_power_EV)
        #     self.plot_power_AC1.append(i_plot_power_AC)
        #
        # lin1_ev = plt.plot(np.arange(len(self.plot_power_EV1)), self.plot_power_EV1, 'cornflowerblue', label="EV")
        # plt.fill_between(np.arange(len(self.plot_power_EV1)),
        #                  self.plot_power_EV1,
        #                  self.plot_load1,
        #                  facecolor='cornflowerblue',  # 填充颜色
        #                  edgecolor='cornflowerblue',  # 边界颜色
        #                  alpha=0.3)
        #
        # plot_power_AC = np.array(self.plot_power_AC1)
        # lin1_ac = plt.plot(np.arange(len(self.plot_power_AC1)), self.plot_power_AC1, 'red', label="AC")
        # plt.fill_between(np.arange(len(self.plot_power_AC1)),
        #                  self.plot_power_AC1,
        #                  self.plot_power_EV1,
        #                  facecolor='red',  # 填充颜色
        #                  edgecolor='red',  # 边界颜色
        #                  alpha=0.2)
        # plot_bes1 = []
        # for i in range(0,96):
        #     if self.plot_action_BES[i]>0:
        #         i_plot_bes = self.plot_action_BES[i]
        #     else:
        #         i_plot_bes = 0
        #     i_plot_bes += self.plot_power_AC1[i]
        #     plot_bes1.append(i_plot_bes)
        # plot_bes2 = []
        # for i in range(0, 96):
        #     if self.plot_action_BES[i] < 0:
        #         i_plot_bes = self.plot_action_BES[i]
        #     else:
        #         i_plot_bes = 0
        #     i_plot_bes += plot_generated[i]
        #     plot_bes2.append(i_plot_bes)
        # # lin2 = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
        # #                 label='BES1')
        # plot_power_bes = np.array(self.plot_action_BES)
        # # lin1_bes = plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, 'grey', label="BES")
        # lin1_bes = plt.plot(np.arange(len(plot_bes1)), plot_bes1, 'slateblue', label="BES1")
        # plt.fill_between(np.arange(len(plot_bes1)),
        #                  plot_bes1,
        #                  self.plot_power_AC1,
        #                  # plot_power_EV - plot_power_EV,
        #                  facecolor='slateblue',  # 填充颜色
        #                  edgecolor='slateblue',  # 边界颜色
        #                  alpha=0.2)
        # lin1_bes = plt.plot(np.arange(len(plot_bes2)), plot_bes2, 'slateblue', label="BES1")
        # plt.fill_between(np.arange(len(plot_bes2)),
        #                  plot_generated,
        #                  plot_bes2,
        #                  # plot_power_EV - plot_power_EV,
        #                  facecolor='slateblue',  # 填充颜色
        #                  edgecolor='slateblue',  # 边界颜色
        #                  alpha=0.2)
        # # plt.fill_between(np.arange(len(self.plot_action_BES)),
        # #                  self.plot_action_BES,
        # #                  self.plot_power_AC1,
        # #                  plot_generated,
        # #                  # plot_power_EV - plot_power_EV,
        # #                  facecolor='grey',  # 填充颜色
        # #                  edgecolor='grey',  # 边界颜色
        # #                  alpha=0.2)
        #
        # ax2 = ax.twinx()
        # # plt.ylim(-2, 2)  # y轴范围设置
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(0, 1)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(0.2)  # 有y轴刻度线间隔
        # ax2.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # buy_price = np.genfromtxt("buy_price.csv", delimiter=',', skip_header=1)  # 微网1单个运行
        # # plt.plot(np.arange(len(buy_price)), buy_price, linewidth=2.0, alpha=1, color='#1f77b4',
        # #          label='buy_price')
        # lin2 = ax2.step(np.arange(len(buy_price)), buy_price, '-.', linewidth=2.0, alpha=1, color='#1f77b4',
        #                 label='Price')
        # ax2.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # # 添加
        #
        # lines = lin1_pv + lin1_load + lin1_ac + lin1_ev + lin1_bes
        # labs = [label.get_label() for label in lines]
        # ax.legend(lines, labs, loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        # plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        # # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # plt.show()

        """Power叠加2"""
        # # 设置图片大小
        # # fig = plt.figure(figsize=(10.0, 7.0))
        # fig = plt.figure(figsize=(18.0, 6.0))
        # fig.patch.set_facecolor('white')  # 设置背景颜色
        # fig.patch.set_alpha(1)  # 设置透明度
        # bwith = 1  # 边框宽度设置为2
        # ax = plt.gca()  # 获取边框
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(-0.5, 2)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        # ax.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # plot_power_EV = np.array(self.plot_power_EV)
        # self.plot_power_EV1 = []
        # self.plot_power_AC1 = []
        # for i in range(0, 96):
        #     i_plot_power_EV = self.plot_power_EV[i]
        #     i_plot_power_AC = self.plot_power_AC[i]
        #     i_plot_power_EV = i_plot_power_EV
        #     i_plot_power_AC = i_plot_power_AC + i_plot_power_EV
        #     self.plot_power_EV1.append(i_plot_power_EV)
        #     self.plot_power_AC1.append(i_plot_power_AC)
        #
        # lin1_ev = plt.plot(np.arange(len(self.plot_power_EV1)), self.plot_power_EV1, 'cornflowerblue', label="EV")
        # plt.fill_between(np.arange(len(self.plot_power_EV1)),
        #                  self.plot_power_EV1,
        #                  standard0,
        #                  facecolor='cornflowerblue',  # 填充颜色
        #                  edgecolor='cornflowerblue',  # 边界颜色
        #                  alpha=0.3)
        #
        # plot_power_AC = np.array(self.plot_power_AC1)
        # lin1_ac = plt.plot(np.arange(len(self.plot_power_AC1)), self.plot_power_AC1, 'red', label="AC")
        # plt.fill_between(np.arange(len(self.plot_power_AC1)),
        #                  self.plot_power_AC1,
        #                  self.plot_power_EV1,
        #                  facecolor='red',  # 填充颜色
        #                  edgecolor='red',  # 边界颜色
        #                  alpha=0.2)
        # plot_bes1 = []
        # for i in range(0, 96):
        #     if self.plot_action_BES[i] > 0:
        #         i_plot_bes = self.plot_action_BES[i]
        #     else:
        #         i_plot_bes = 0
        #     i_plot_bes += self.plot_power_AC1[i]
        #     plot_bes1.append(i_plot_bes)
        # plot_bes2 = []
        # for i in range(0, 96):
        #     if self.plot_action_BES[i] < 0:
        #         i_plot_bes = self.plot_action_BES[i]
        #     else:
        #         i_plot_bes = 0
        #     # i_plot_bes += plot_generated[i]
        #     plot_bes2.append(i_plot_bes)
        # # lin2 = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
        # #                 label='BES1')
        # plot_power_bes = np.array(self.plot_action_BES)
        # # lin1_bes = plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, 'grey', label="BES")
        # lin1_bes = plt.plot(np.arange(len(plot_bes1)), plot_bes1, 'slateblue', label="BES1")
        # plt.fill_between(np.arange(len(plot_bes1)),
        #                  plot_bes1,
        #                  self.plot_power_AC1,
        #                  # plot_power_EV - plot_power_EV,
        #                  facecolor='slateblue',  # 填充颜色
        #                  edgecolor='slateblue',  # 边界颜色
        #                  alpha=0.2)
        # lin1_bes = plt.plot(np.arange(len(plot_bes2)), plot_bes2, 'slateblue', label="BES1")
        # plt.fill_between(np.arange(len(plot_bes2)),
        #                  standard0,
        #                  plot_bes2,
        #                  # plot_power_EV - plot_power_EV,
        #                  facecolor='slateblue',  # 填充颜色
        #                  edgecolor='slateblue',  # 边界颜色
        #                  alpha=0.2)
        # # plt.fill_between(np.arange(len(self.plot_action_BES)),
        # #                  self.plot_action_BES,
        # #                  self.plot_power_AC1,
        # #                  plot_generated,
        # #                  # plot_power_EV - plot_power_EV,
        # #                  facecolor='grey',  # 填充颜色
        # #                  edgecolor='grey',  # 边界颜色
        # #                  alpha=0.2)
        #
        # ax2 = ax.twinx()
        # # plt.ylim(-2, 2)  # y轴范围设置
        # # 设置边框
        # ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # ax.spines['left'].set_linewidth(bwith)  # 图框左边
        # ax.spines['top'].set_linewidth(bwith)  # 图框上边
        # ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # # 显示网格线
        # # plt.grid(True)
        # # 同时设置
        # # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # 单独设置x和y
        # # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        #
        # plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.yticks(fontproperties='Times New Roman', size=18)
        # # 刻度线的大小长短粗细
        # plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # # 不显示刻度标签
        # # ax.axes.xaxis.set_ticklabels([])
        # # ax.axes.yaxis.set_ticklabels([])
        # plt.xlim(0, 96)  # x轴范围设置
        # plt.ylim(0, 1)  # y轴范围设置
        # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        # y_major_locator = MultipleLocator(0.2)  # 有y轴刻度线间隔
        # ax2.yaxis.set_major_locator(y_major_locator)
        # ax.xaxis.set_major_locator(x_major_locator)
        # buy_price = np.genfromtxt("buy_price.csv", delimiter=',', skip_header=1)  # 微网1单个运行
        # # plt.plot(np.arange(len(buy_price)), buy_price, linewidth=2.0, alpha=1, color='#1f77b4',
        # #          label='buy_price')
        # lin2 = ax2.step(np.arange(len(buy_price)), buy_price, '-.', linewidth=2.0, alpha=1, color='#1f77b4',
        #                 label='Price')
        # ax2.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # ax.set_ylabel("Power/kWh", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # # 添加
        #
        # lines = lin1_ac + lin1_ev + lin1_bes
        # labs = [label.get_label() for label in lines]
        # ax.legend(lines, labs, loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        # plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        # # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # plt.show()

        # kW

        """Power叠加 kW"""
        # plot_load2 = []
        # plot_generated2 = []
        # self.plot_power_EV2 = []
        # self.plot_power_AC2 = []
        # plot_bes11 = []
        # plot_bes22 = []
        # for i in range(0,96):
        #     i_plot_load = plot_load[i]/0.25
        #     plot_load2.append(i_plot_load)
        #     i_plot_generated = plot_generated[i]/0.25
        #     plot_generated2.append(i_plot_generated)
        #     i_bes = self.plot_power_EV1[i]/0.25
        #     self.plot_power_EV2.append(i_bes)
        #     i_ac = self.plot_power_AC1[i] / 0.25
        #     self.plot_power_AC2.append(i_ac)
        #     i_bes1 = plot_bes1[i]/0.25
        #     plot_bes11.append(i_bes1)
        #     i_bes2 = plot_bes2[i]/0.25
        #     plot_bes22.append(i_bes2)

        # 设置图片大小
        # 设置图片大小
        # fig = plt.figure(figsize=(10.0, 7.0))
        fig = plt.figure(figsize=(18.0, 6.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 96)  # x轴范围设置
        plt.ylim(-6, 20)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(5)  # 有y轴刻度线间隔
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)
        # 添加
        plot_generated1 = np.genfromtxt("PV_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        plot_generated = []
        standard0 = []
        for i in range(0, 96):
            # for i_plot_generated in plot_generated1:
            i_plot_generated = -plot_generated1[i]/0.25
            plot_generated.append(i_plot_generated)
            standard0.append(0)
        plot_load1 = np.genfromtxt("load_MG1.csv", delimiter=',', skip_header=1, usecols=[-1])
        plot_load = []
        for i in range(0, 96):
            # for i_plot_generated in plot_generated1:
            i_plot_load = plot_load1[i]/0.25
            plot_load.append(i_plot_load)
        # ax1 = fig.subplots()
        # ax2 = ax1.twinx()
        lin1_load = plt.plot(np.arange(len(plot_load)), plot_load, 'orange', label="Load")
        plt.fill_between(np.arange(len(plot_load)),
                         plot_load,
                         # plot_load - plot_load,
                         standard0,
                         facecolor='orange',  # 填充颜色
                         edgecolor='orange',  # 边界颜色
                         alpha=0.2)
        lin1_pv = plt.plot(np.arange(len(plot_generated)), plot_generated, 'green', label="PV")
        plt.fill_between(np.arange(len(plot_generated)),
                         plot_generated,
                         # plot_load - plot_load,
                         standard0,
                         facecolor='green',  # 填充颜色
                         edgecolor='green',  # 边界颜色
                         alpha=0.3)
        plot_power_EV = np.array(self.plot_power_EV)
        self.plot_power_EV1 = []
        self.plot_load1 = []
        self.plot_power_AC1 = []
        for i in range(0, 96):
            i_plot_power_EV = self.plot_power_EV[i]/0.25
            i_plot_load = plot_load[i]
            i_plot_power_AC = self.plot_power_AC[i]/0.25
            i_plot_power_EV = i_plot_power_EV + i_plot_load
            i_plot_power_AC = i_plot_power_AC + i_plot_power_EV
            self.plot_load1.append(i_plot_load)
            self.plot_power_EV1.append(i_plot_power_EV)
            self.plot_power_AC1.append(i_plot_power_AC)

        lin1_ev = plt.plot(np.arange(len(self.plot_power_EV1)), self.plot_power_EV1, 'cornflowerblue', label="EV")
        plt.fill_between(np.arange(len(self.plot_power_EV1)),
                         self.plot_power_EV1,
                         self.plot_load1,
                         facecolor='cornflowerblue',  # 填充颜色
                         edgecolor='cornflowerblue',  # 边界颜色
                         alpha=0.3)

        plot_power_AC = np.array(self.plot_power_AC1)
        lin1_ac = plt.plot(np.arange(len(self.plot_power_AC1)), self.plot_power_AC1, 'red', label="AC")
        plt.fill_between(np.arange(len(self.plot_power_AC1)),
                         self.plot_power_AC1,
                         self.plot_power_EV1,
                         facecolor='red',  # 填充颜色
                         edgecolor='red',  # 边界颜色
                         alpha=0.2)
        plot_bes1 = []
        for i in range(0, 96):
            if self.plot_action_BES[i] > 0:
                i_plot_bes = (self.plot_action_BES[i]/0.25)
            else:
                i_plot_bes = 0
            i_plot_bes += self.plot_power_AC1[i]
            plot_bes1.append(i_plot_bes)
        plot_bes2 = []
        for i in range(0, 96):
            if self.plot_action_BES[i] < 0:
                i_plot_bes = (self.plot_action_BES[i]/0.25)
            else:
                i_plot_bes = 0
            i_plot_bes += plot_generated[i]
            plot_bes2.append(i_plot_bes)
        # lin2 = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
        #                 label='BES1')
        plot_power_bes = np.array(self.plot_action_BES)
        # lin1_bes = plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, 'grey', label="BES")
        lin1_bes = plt.plot(np.arange(len(plot_bes1)), plot_bes1, 'slateblue', label="BES1")
        plt.fill_between(np.arange(len(plot_bes1)),
                         plot_bes1,
                         self.plot_power_AC1,
                         # plot_power_EV - plot_power_EV,
                         facecolor='slateblue',  # 填充颜色
                         edgecolor='slateblue',  # 边界颜色
                         alpha=0.2)
        lin1_bes = plt.plot(np.arange(len(plot_bes2)), plot_bes2, 'slateblue', label="BES1")
        plt.fill_between(np.arange(len(plot_bes2)),
                         plot_generated,
                         plot_bes2,
                         # plot_power_EV - plot_power_EV,
                         facecolor='slateblue',  # 填充颜色
                         edgecolor='slateblue',  # 边界颜色
                         alpha=0.2)
        # plt.fill_between(np.arange(len(self.plot_action_BES)),
        #                  self.plot_action_BES,
        #                  self.plot_power_AC1,
        #                  plot_generated,
        #                  # plot_power_EV - plot_power_EV,
        #                  facecolor='grey',  # 填充颜色
        #                  edgecolor='grey',  # 边界颜色
        #                  alpha=0.2)

        ax2 = ax.twinx()
        # plt.ylim(-2, 2)  # y轴范围设置
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 96)  # x轴范围设置
        plt.ylim(0, 1)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(0.2)  # 有y轴刻度线间隔
        ax2.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)
        buy_price = np.genfromtxt("buy_price.csv", delimiter=',', skip_header=1)  # 微网1单个运行
        # plt.plot(np.arange(len(buy_price)), buy_price, linewidth=2.0, alpha=1, color='#1f77b4',
        #          label='buy_price')
        lin2 = ax2.step(np.arange(len(buy_price)), buy_price, '-.', linewidth=2.0, alpha=1, color='#1f77b4',
                        label='Price')
        ax2.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        ax.set_ylabel("Power/kW", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # 添加

        lines = lin1_pv + lin1_load + lin1_ac + lin1_ev + lin1_bes
        labs = [label.get_label() for label in lines]
        ax.legend(lines, labs, loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()

        """Power叠加2kW"""
        # 设置图片大小
        # fig = plt.figure(figsize=(10.0, 7.0))
        fig = plt.figure(figsize=(18.0, 6.0))
        fig.patch.set_facecolor('white')  # 设置背景颜色
        fig.patch.set_alpha(1)  # 设置透明度
        bwith = 1  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 96)  # x轴范围设置
        plt.ylim(-1, 6)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(1)  # 有y轴刻度线间隔
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)
        plot_power_EV = np.array(self.plot_power_EV)
        self.plot_power_EV1 = []
        self.plot_power_AC1 = []
        for i in range(0, 96):
            i_plot_power_EV = self.plot_power_EV[i]/0.25
            i_plot_power_AC = self.plot_power_AC[i]/0.25
            i_plot_power_EV = i_plot_power_EV
            i_plot_power_AC = i_plot_power_AC + i_plot_power_EV
            self.plot_power_EV1.append(i_plot_power_EV)
            self.plot_power_AC1.append(i_plot_power_AC)

        lin1_ev = plt.plot(np.arange(len(self.plot_power_EV1)), self.plot_power_EV1, 'cornflowerblue', label="EV")
        plt.fill_between(np.arange(len(self.plot_power_EV1)),
                         self.plot_power_EV1,
                         standard0,
                         facecolor='cornflowerblue',  # 填充颜色
                         edgecolor='cornflowerblue',  # 边界颜色
                         alpha=0.3)

        plot_power_AC = np.array(self.plot_power_AC1)
        lin1_ac = plt.plot(np.arange(len(self.plot_power_AC1)), self.plot_power_AC1, 'red', label="AC")
        plt.fill_between(np.arange(len(self.plot_power_AC1)),
                         self.plot_power_AC1,
                         self.plot_power_EV1,
                         facecolor='red',  # 填充颜色
                         edgecolor='red',  # 边界颜色
                         alpha=0.2)
        plot_bes1 = []
        for i in range(0, 96):
            if self.plot_action_BES[i] > 0:
                i_plot_bes = self.plot_action_BES[i]/0.25
            else:
                i_plot_bes = 0
            i_plot_bes += self.plot_power_AC1[i]
            plot_bes1.append(i_plot_bes)
        plot_bes2 = []
        for i in range(0, 96):
            if self.plot_action_BES[i] < 0:
                i_plot_bes = self.plot_action_BES[i]/0.25
            else:
                i_plot_bes = 0
            # i_plot_bes += plot_generated[i]
            plot_bes2.append(i_plot_bes)
        # lin2 = ax2.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, '-.', linewidth=2.0, alpha=1,
        #                 label='BES1')
        plot_power_bes = np.array(self.plot_action_BES)
        # lin1_bes = plt.plot(np.arange(len(self.plot_action_BES)), self.plot_action_BES, 'grey', label="BES")
        lin1_bes = plt.plot(np.arange(len(plot_bes1)), plot_bes1, 'slateblue', label="BES1")
        plt.fill_between(np.arange(len(plot_bes1)),
                         plot_bes1,
                         self.plot_power_AC1,
                         # plot_power_EV - plot_power_EV,
                         facecolor='slateblue',  # 填充颜色
                         edgecolor='slateblue',  # 边界颜色
                         alpha=0.2)
        lin1_bes = plt.plot(np.arange(len(plot_bes2)), plot_bes2, 'slateblue', label="BES1")
        plt.fill_between(np.arange(len(plot_bes2)),
                         standard0,
                         plot_bes2,
                         # plot_power_EV - plot_power_EV,
                         facecolor='slateblue',  # 填充颜色
                         edgecolor='slateblue',  # 边界颜色
                         alpha=0.2)
        # plt.fill_between(np.arange(len(self.plot_action_BES)),
        #                  self.plot_action_BES,
        #                  self.plot_power_AC1,
        #                  plot_generated,
        #                  # plot_power_EV - plot_power_EV,
        #                  facecolor='grey',  # 填充颜色
        #                  edgecolor='grey',  # 边界颜色
        #                  alpha=0.2)

        ax2 = ax.twinx()
        # plt.ylim(-2, 2)  # y轴范围设置
        # 设置边框
        ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
        ax.spines['left'].set_linewidth(bwith)  # 图框左边
        ax.spines['top'].set_linewidth(bwith)  # 图框上边
        ax.spines['right'].set_linewidth(bwith)  # 图框右边
        # 显示网格线
        # plt.grid(True)
        # 同时设置
        # plt.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # 单独设置x和y
        # ax.xaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)
        # ax.yaxis.grid(color='lightgray', linewidth=0.5, alpha=0.8)

        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)
        # 刻度线的大小长短粗细
        plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)
        # 不显示刻度标签
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.xlim(0, 96)  # x轴范围设置
        plt.ylim(0, 1)  # y轴范围设置
        x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
        y_major_locator = MultipleLocator(0.2)  # 有y轴刻度线间隔
        ax2.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)
        buy_price = np.genfromtxt("buy_price.csv", delimiter=',', skip_header=1)  # 微网1单个运行
        # plt.plot(np.arange(len(buy_price)), buy_price, linewidth=2.0, alpha=1, color='#1f77b4',
        #          label='buy_price')
        lin2 = ax2.step(np.arange(len(buy_price)), buy_price, '-.', linewidth=2.0, alpha=1, color='#1f77b4',
                        label='Price')
        ax2.set_ylabel("Price", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        ax.set_ylabel("Power/kW", fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        # 添加

        lines = lin1_ac + lin1_ev + lin1_bes
        labs = [label.get_label() for label in lines]
        ax.legend(lines, labs, loc='best', prop={'family': 'Times New Roman', 'size': 20})
        # ax.legend(loc='best', prop={'family': 'Times New Roman', 'size': 20})  # 显示图例，内容为上面的label
        plt.xlabel('Time/15min', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)  # labelpad为标题距离刻度线范围
        # plt.ylabel('Power/kWh', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=5)
        plt.show()


        print("r1", self.r1)
        print("r2", self.r2)
        print("r3", self.r3)
        print("r4", self.r4)
        print("r", self.r)

'''Hyperparameter Setting'''  # 超参数设置
parser = argparse.ArgumentParser()
# parser.add_argument('--EnvIdex', type=int, default=3, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--EnvIdex', type=int, default=0, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=384000, help='which model to load')

parser.add_argument('--seed', type=int,  default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=96, help='lenth of long trajectory')
parser.add_argument('--distnum', type=int, default=0, help='0:Beta ; 1:GS_ms  ;  2: GS_m')
# parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
# parser.add_argument('--Max_train_steps', type=int, default=5e6, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

# parser.add_argument('--Max_train_steps', type=int, default=5e5, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

# parser.add_argument('--Max_train_steps', type=int, default=5e4, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=5e4, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=2048, help='Model evaluating interval, in steps.')
# parser.add_argument('--Max_train_steps', type=int, default=2e5, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=2e5, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=95, help='Model evaluating interval, in steps.')
# parser.add_argument('--Max_train_steps', type=int, default=9600, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=9600, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=96, help='Model evaluating interval, in steps.')
# parser.add_argument('--Max_train_steps', type=int, default=384000, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=384000, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=96, help='Model evaluating interval, in steps.')
parser.add_argument('--Max_train_steps', type=int, default=288000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=288000, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=96, help='Model evaluating interval, in steps.')

# parser.add_argument('--Max_train_steps', type=int, default=480000, help='Max training steps')
# parser.add_argument('--save_interval', type=int, default=480000, help='Model saving interval, in steps.')
# parser.add_argument('--eval_interval', type=int, default=96, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
# print(opt)

def Action_adapter(a,max_action): # 动作适配
    # from [0,1] to [-max,max]
    # return  2*(a-0.5)*max_action
    return a

def Reward_adapter(r, EnvIdex): # 奖励适配
    # For BipedalWalker 双足行走者环境
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: r = -1
    # For Pendulum-v0 钟摆环境
    elif EnvIdex == 3:
        r = (r + 8) / 8
    return r

def evaluate_policy(env, model, render, steps_per_epoch, max_action, EnvIdex): # 评价策略
    scores = 0
    turns = 3
    for j in range(turns):
        s, done, ep_r, steps = env.reset(main.total_steps), False, 0, 0
        while not (done or (steps >= steps_per_epoch)):
            # Take deterministic actions at test time 在测试时间段采取确定动作
            a, logprob_a = model.evaluate(s)  # 动作，动作概率
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max] 动作适配
            # print("`````act`````",act)
            # act[1] = env.change_a((act[1]))
            s_prime, r, done, info = env.step(act)
            # r = Reward_adapter(r, EnvIdex)

            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores/turns

plot1 = PLOT()
all_ep_r = []
plot_score=[]
from env1 import *
env =MicroGridEnv()
eval_env = MicroGridEnv()
class main():
    def __init__(self):
        self.write = opt.write   #Use SummaryWriter to record the training.
        self.render = opt.render

        self.EnvName = ['Microgrid','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v1','Humanoid-v2','HalfCheetah-v2']
        BriefEnvName = ['MG', 'BWHv3', 'Lch_Cv2', 'PV0', 'Humanv2', 'HCv2']
        Env_With_Dead = [True, True, True, False, True, False]
        self.EnvIdex = opt.EnvIdex
        env_with_Dead = Env_With_Dead[self.EnvIdex]  #Env like 'LunarLanderContinuous-v2' is with Dead Signal. Important!
        state_dim = 3
        action_dim = 3
        self.max_action = 1
        self.max_steps = 96
        # print('Env:',EnvName[EnvIdex],'  state_dim:',state_dim,'  action_dim:',action_dim,
              # '  max_a:',max_action,'  min_a:',0, 'max_steps', max_steps)
        self.T_horizon = opt.T_horizon  #lenth of long trajectory


        Dist = ['Beta', 'GS_ms', 'GS_m'] #type of probility distribution
        distnum = opt.distnum

        self.Max_train_steps = opt.Max_train_steps
        self.save_interval = opt.save_interval#in steps
        self.eval_interval = opt.eval_interval#in steps


        self.random_seed = opt.seed
        # print("Random Seed: {}".format(random_seed))
        torch.manual_seed(self.random_seed)
        # env.seed(self.random_seed)
        # eval_env.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.write:
            timenow = str(datetime.now())[0:-10]
            timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
            writepath = 'runs1/{}'.format(BriefEnvName[self.EnvIdex]) + timenow
            if os.path.exists(writepath): shutil.rmtree(writepath)
            self.writer = SummaryWriter(log_dir=writepath)

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "env_with_Dead":env_with_Dead,
            "gamma": opt.gamma,
            "lambd": opt.lambd,     #For GAE
            "clip_rate": opt.clip_rate,  #0.2
            "K_epochs": opt.K_epochs,
            "net_width": opt.net_width,
            "a_lr": opt.a_lr,
            "c_lr": opt.c_lr,
            "dist": Dist[distnum],
            "l2_reg": opt.l2_reg,   #L2 regulization for Critic
            "a_optim_batch_size":opt.a_optim_batch_size,
            "c_optim_batch_size": opt.c_optim_batch_size,
            "entropy_coef":opt.entropy_coef, #Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
            "entropy_coef_decay":opt.entropy_coef_decay
        }
        # if Dist[distnum] == 'Beta' :
        #     kwargs["a_lr"] *= 2 #Beta dist need large lr|maybe
        #     kwargs["c_lr"] *= 4  # Beta dist need large lr|maybe

        if not os.path.exists('model1'): os.mkdir('model1')
        self.model = PPO(**kwargs)
        if opt.Loadmodel: self.model.load(opt.ModelIdex)

        self.traj_lenth = 0
        self.total_steps = 0
    def main(self):
        # print("step",self.steps)
        # while total_steps < Max_train_steps:
        #     s, done, steps, ep_r = env.reset(), False, 0, 0
        '''Interact & trian'''
        # while not self.done:
        if self.steps == self.max_steps:  # +
            self.s, self.done, self.steps, self.ep_r = env.reset(), False, 0, 0  # 添加
            self.steps = 0
        # print("``````step`````", self.steps)
        # print("self.s", self.s)
        self.traj_lenth += 1
        self.steps += 1
        if self.render:
            env.render()
            a, logprob_a = self.model.evaluate(self.s)
        else:
            a, logprob_a = self.model.select_action(self.s)
        # print("step,step", self.steps, self.s)
        act = Action_adapter(a, self.max_action)  # [0,1] to [-max,max]
        s_prime, r, self.done, info = env.step(act)
        r = Reward_adapter(r, self.EnvIdex)

        """plot"""
        # if total_steps >= 499890 and total_steps <= 499984:
        # if total_steps >= 49875 and total_steps <= 49969:
        # if self.total_steps >= 199880 and self.total_steps <= 199974:
        if int(self.total_steps /96) == 2999:
            print("````step``````", self.steps)
            plot1.record_MG()

        '''distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax'''
        '''dw for TD_target and Adv; done for GAE'''
        if self.done and self.steps != self.max_steps:
            dw = True
            #still have exception: dead or win at _max_episode_steps will not be regard as dw.
            #Thus, decide dw according to reward signal of each game is better.  dw = done_adapter(r)
        else:
            dw = False

        self.model.put_data((self.s, a, r, s_prime, logprob_a, self.done, dw))
        self.s = s_prime
        self.ep_r += r

        '''update if its time'''
        # if not render:
        # if self.traj_lenth % self.T_horizon == 0:
        if main.traj_lenth % main.T_horizon == 0:
            self.model.train()
        # print("1",self.critic_C1_weight.data[0])
        # print("2", critic_C1_weight1.data[0])
            self.traj_lenth = 0

        '''record & log'''
        if self.total_steps % self.eval_interval == 0 and self.total_steps != 0:
            score = evaluate_policy(eval_env, self.model, False, self.max_steps, self.max_action,
                                    self.EnvIdex)  # 重点evaluate_policy
            if self.write:
                self.writer.add_scalar('ep_r_insteps', score, global_step=self.total_steps)
            # print('EnvName1:', self.EnvName[self.EnvIdex], 'seed:', self.random_seed,
            #       'steps: {}k'.format(int(self.total_steps / 1000)), 'score:', score)
            print('EnvName1:', self.EnvName[self.EnvIdex], 'seed:', self.random_seed,
                  'steps: {}'.format(int(self.total_steps/96)), 'score:', score)
            plot_score.append(score)
            np.savetxt("MG1_score.csv", plot_score, delimiter=",", fmt="%.8f")
        self.total_steps += 1

        '''save model'''
        if self.total_steps % self.save_interval == 0:
            self.model.save(self.total_steps)
        # print("done1",done)

    def reset(self):
        self.s, self.done, self.steps, self.ep_r = env.reset(self.total_steps), False, 0, 0

    def done(self):
        return self.done

    def main_plot(self):
        plot1.plot_MG()

main = main()
if __name__ == '__main__':
    while main.total_steps < main.Max_train_steps:
        main.reset()
        while not main.done:
            main.main()
    env.close()

    np.savetxt("MG1_score_yuan.csv", plot_score, delimiter=",", fmt="%.8f")
    plot1.plot_MG()