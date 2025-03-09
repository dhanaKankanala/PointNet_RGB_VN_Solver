import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

#plt.rcParams['font.family'] = 'Times New Roman'

visualization = '0p35_sp_uniform_spiral'


if visualization == '1_uniform_ellipse':
    title = 'Circular'
    #color = 'red'
if visualization == '0p35_sp_uniform_spiral':
    title = 'Spiral'
    #color = 'blue'
if visualization == 'uniform_random':
    title = 'Random'
    #color = 'darkgreen'
exp=2
if exp==1:
    tn_s = 160
    tt_s = 500
    val_s = 40
    label = '80-20'
if exp==2:
    tn_s = 80
    tt_s= 500
    val_s = 20
    label = '160-40'

model_size = 'small'
test_size = 'small'

if visualization == 'uniform_random':
    place= 'upper right'
    col=2
elif visualization == '1_uniform_ellipse' and exp == 2:
    place = 'upper right'
    col = 3
else:
    place = 'lower right'
    col = 1

file_1 = 'result_' + visualization + '_' + model_size + '_' + test_size + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s)

file_name_1 = pd.read_csv(file_1+'.csv')



plt.plot(file_name_1['F1_3'], linestyle='--', marker='x', label='s=3', color='darkgreen', markersize=10, linewidth=2.5)
plt.plot(file_name_1['F1_7'], linestyle='-.', marker='*', label='s=7', markersize=10, linewidth=2.5)
plt.plot(file_name_1['F1_11'],  linestyle='--', marker='^', label='s=11', markersize=10, linewidth=2.5)
plt.plot(file_name_1['F1_13'], linestyle='-', marker='D',  label='s=13', markersize=10, linewidth=2.5)
plt.plot(file_name_1['F1_29'], linestyle='--', marker='s', color='blue', markersize=10, label='s=29', linewidth=2.5)
# Epoch values
epochs_1 = range(1,len(file_name_1)+1)

plt.tight_layout()


ytick_values = [i / 11 for i in range(1, 10)]
plt.yticks(ytick_values)

plt.legend(loc=place, fontsize=20, facecolor='none', framealpha=0, ncol=col)
plt.xticks(fontsize=22)
xtick_values = epochs_1[::4]
plt.xticks(xtick_values)
plt.yticks([0, 0.3, 0.7, 1.0])
plt.yticks(fontsize=22)
plt.savefig(visualization + '_exp_'+ str(exp) + '_all_seeds.pdf', dpi=500, bbox_inches='tight')
plt.show()
