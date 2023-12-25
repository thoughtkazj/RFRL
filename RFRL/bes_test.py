# import numpy as np
# import matplotlib.pyplot as plt
#
# def R_t_BES(SOC):
#     if 0 < SOC < 0.2:
#         return 2 * np.exp(0.9) - np.exp(-np.abs(0.2 - SOC))
#     elif 0.2 < SOC < 0.9:
#         return 2 * np.exp(0.9) - 2 * np.exp(SOC)
#     elif 0.9 < SOC < 1:
#         return 2 * np.exp(0.9) - np.exp(-np.abs(0.9 - SOC))
#     else:
#         return None
#
# # 生成SOC的数据
# SOC_values = np.linspace(0, 1, 1000)
#
# # 计算对应的R_t_BES值
# R_values = [R_t_BES(soc) for soc in SOC_values]
#
# # 绘图
# plt.plot(SOC_values, R_values, label=r'$R_t^{BES}$')
# plt.xlabel('SOC (State of Charge)')
# plt.ylabel(r'$R_t^{BES}$')
# plt.title('Graph of $R_t^{BES}$')
# plt.legend()
# plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

def C_t_BES_i(SOC):
    if 0 < SOC < 0.2:
        return 2 * np.exp(0.9) - np.exp(-np.abs(0.2 - SOC))
    elif 0.2 < SOC < 0.9:
        return 2 * np.exp(0.9) - 2 * np.exp(SOC)
    elif 0.9 < SOC < 1:
        return 2 * np.exp(0.9) - np.exp(-np.abs(0.9 - SOC))
    else:
        return None

# 生成SOC的数据
SOC_values = np.linspace(0, 1, 1000)

# 计算对应的C_t_BES_i值
C_values = [C_t_BES_i(soc) for soc in SOC_values]

# 绘图
plt.plot(SOC_values, C_values, label=r'$C_t^{BES_i}$')
plt.xlabel('SOC (State of Charge)')
plt.ylabel(r'$C_t^{BES_i}$')
plt.title('Graph of $C_t^{BES_i}$')
plt.legend()
plt.grid(True)
plt.show()

