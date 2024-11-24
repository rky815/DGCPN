# -*- coding: gbk -*-
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator

# ����ȫ������������С
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.labelsize'] = 18  # x��y��ǩ�������С
plt.rcParams['xtick.labelsize'] = 20  # x��̶ȵ������С
plt.rcParams['ytick.labelsize'] = 20  # y��̶ȵ������С
plt.rcParams['legend.fontsize'] = 20  # ͼ���������С

# �������ݿ�ܣ���������ָ��MSE��R2
df = pd.DataFrame({
    'Model': ['(EMD)-SVR', '(EMD)-LSTM', '(EMD)-GRU', '(EMD)-TCN', '(EMD)-TAE',
              '(EEMD)-SVR', '(EEMD)-LSTM', '(EEMD)-GRU', '(EEMD)-TCN', '(EEMD)-TAE',
              '(CEEMDAN)-SVR', '(CEEMDAN)-LSTM', '(CEEMDAN)-GRU', '(CEEMDAN)-TCN', '(CEEMDAN)-TAE'] * 2,
    'Metric': ['MAPE']*15 + ['R2']*15,
    'Value': [
        2.5388, 1.5027, 1.5325, 1.1729, 0.6752,
        2.6684, 1.4482, 1.5087, 1.1985, 0.6505,
        2.4700, 1.1754, 1.3681, 1.1227, 0.5323,
        0.3000, 0.6072, 0.5915, 0.7607, 0.9207,
        0.2268, 0.6352, 0.6040, 0.7501, 0.9264,
        0.3375, 0.7597, 0.6744, 0.7807, 0.9507
    ]
})

# ��������ͼ
plt.figure(figsize=(15, 10))
barplot = sns.barplot(x="Model", y="Value", hue="Metric", data=df, palette=["#1B3C73", "#FF407D"],
                      edgecolor="black", linewidth=1.5)

# ȥ��ͼ������
plt.legend().set_title(None)

# ���ñ�������ǩ
plt.title('')
plt.xlabel('')
plt.ylabel('')

# ��תX���ǩ�Ա����Ǹ������Ķ�
# ��תX���ǩ�Ա����Ǹ������Ķ�
x_ticks = barplot.get_xticks()
barplot.xaxis.set_major_locator(FixedLocator(x_ticks))
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
# # ������״�Ŀ��
# for patch in barplot.patches:
#     current_width = patch.get_width()
#     new_width = current_width * 0.6
#     patch.set_width(new_width)
#     # We also recenter the bar
#     patch.set_x(patch.get_x() + current_width * 0.2)

# ȥ��������
plt.grid(False)

# չʾͼ��
plt.tight_layout()
plt.show()


