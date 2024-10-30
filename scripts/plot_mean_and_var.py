import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

color_group = {
    'D2DSeen': 'blue',
    'D2DUnseen': 'orange',
    'D2DH.M.': 'green',
    'I2ISeen': 'blue',
    'I2IUnseen': 'orange',
    'I2IH.M.': 'green',
    'I2DSeen': 'blue',
    'I2DUnseen': 'orange',
    'I2DH.M.': 'green',
}
color_group = {
    'D2DSeen': '#1F77B4',
    'D2DUnseen': '#1F77B4',
    'D2DH.M.': '#1F77B4',
    'I2ISeen': '#FF7F0E',
    'I2IUnseen': '#FF7F0E',
    'I2IH.M.': '#FF7F0E',
    'I2DSeen': '#2CA02C',
    'I2DUnseen': '#2CA02C',
    'I2DH.M.': '#2CA02C',
}
filler_group = {
    'D2DSeen': '#AEC7E8',
    'D2DUnseen': '#AEC7E8',
    'D2DH.M.': '#AEC7E8',
    'I2ISeen': '#FFBB78',
    'I2IUnseen': '#FFBB78',
    'I2IH.M.': '#FFBB78',
    'I2DSeen': '#98DF8A',
    'I2DUnseen': '#98DF8A',
    'I2DH.M.': '#98DF8A',
}

marker_group = {
    'D2DSeen': 'o',
    'D2DUnseen': 'o',
    'D2DH.M.': 'o',
    'I2ISeen': 'X',
    'I2IUnseen': 'X',
    'I2IH.M.': 'X',
    'I2DSeen': 's',
    'I2DUnseen': 's',
    'I2DH.M.': 's',
}

linestyle_group = {
    'D2DSeen': (),
    'D2DUnseen': (2,1),
    'D2DH.M.': (6,2),
    'I2ISeen': (),
    'I2IUnseen': (2,1),
    'I2IH.M.': (6,2),
    'I2DSeen': (),
    'I2DUnseen': (2,1),
    'I2DH.M.': (6,2),
}

def get_df(accuracy, variance):
    data = {
        # 'Taxonomy Level': np.tile(np.arange(1, 11), 4),
        'Taxonomy Level': np.tile(['Order', 'Family', 'genus', 'species'], 9),
        'Accuracy': np.array(accuracy),
        'Variance': np.array(variance),
        # 'Model': np.tile(
        #     np.repeat(['Seen', 'Unseen', 'H.M.'], 4), 3),
        'Model': np.repeat(['D2DSeen', 'D2DUnseen', 'D2DH.M.', 'I2ISeen', 'I2IUnseen', 'I2IH.M.', 'I2DSeen', 'I2DUnseen', 'I2DH.M.'], 4)
    }
    df = pd.DataFrame(data)
    df['Lower'] = df['Accuracy'] - df['Variance']
    df['Upper'] = df['Accuracy'] + df['Variance']

    return df


accuracy = [100.0, 99.8, 98.775, 96.875, 100.0, 97.32499999999999, 95.575, 90.67500000000001, 100.0, 98.54506065178806, 97.146972761652, 93.67126862579032, 99.32499999999999, 92.65, 75.4, 61.10000000000001, 97.475, 80.825, 62.95, 46.55, 98.38753289997655, 86.32951826278115, 68.6130028645146, 52.840585576262015, 99.325, 89.025, 68.025, 51.1, 76.9, 53.0, 24.2, 9.9, 86.6733611035842, 66.43181525814744, 35.69294519105148, 16.583455835874364]
variance = [0.0, 0.3464101615137742, 0.1479019945774916, 0.22776083947860584, 0.0, 0.6259992012774468, 0.7529110173187779, 0.6495190528383258, 0.0, 0.2730708775063198, 0.38132909640283286, 0.37204889720025897, 0.4205650960315187, 0.37749172176353907, 0.5477225575051674, 0.7141428428542824, 1.3754544703478955, 0.9522998477370457, 0.40311288741492834, 0.3570714214271426, 0.8291139880588726, 0.4273029230979596, 0.3070976722631344, 0.4462885613206655, 0.2586020108197161, 1.1924240017711831, 0.7013380069552777, 0.9513148795220226, 1.95320249846246, 1.905255888325764, 0.8154753215150042, 0.30822070014844916, 1.2871008113357687, 1.7671706372908198, 0.9231492851207391, 0.4336994061987346]

macro_df = get_df(accuracy, variance)

accuracy = [100.0, 100.0, 99.425, 98.57499999999999, 100.0, 99.675, 98.275, 96.575, 100.0, 99.83723079609396, 98.84648364954111, 97.56458285206428, 99.7, 96.55000000000001, 89.65, 80.45, 99.57499999999999, 94.35, 84.225, 74.225, 99.6374560837429, 95.4373232023884, 86.85262884296742, 77.21193778908567, 99.7, 96.525, 86.525, 73.25, 99.075, 86.55, 59.14999999999999, 38.6, 99.38650008977955, 91.26545827428643, 70.26435240859584, 50.554237079090946]
variance = [0.0, 0.0, 0.08291561975888564, 0.1479019945774874, 0.0, 0.043301270189225624, 0.19202864369671585, 0.24874685927665263, 0.0, 0.021726613060487156, 0.07200622162249286, 0.1613638418998894, 0.0, 0.04999999999999716, 0.165831239517767, 0.3201562118716442, 0.04330127018921947, 0.05000000000000426, 0.35619517121937316, 0.19202864369671585, 0.02168324692624216, 0.05000664054354331, 0.24752564573697108, 0.20784490932325048, 0.07071067811865576, 0.22776083947860973, 0.39607448794387207, 0.11180339887499267, 0.08291561975888564, 0.4769696007084729, 0.3905124837953324, 0.6745368781616017, 0.06500744489023505, 0.35853621072516023, 0.3123780894522774, 0.5726629599595011]

micro_df = get_df(accuracy, variance)

# 模拟数据

# 为了模拟 variance，我们可以创建一个区间，在 Accuracy 基础上加减 variance
# sns.set(font_scale=1.1)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
# plt.figure(figsize=(10, 6))
sns.lineplot(data=macro_df, x='Taxonomy Level', y='Accuracy', hue='Model', palette=color_group, style='Model', markers=False, dashes=False, ax=ax1)
# sns.move_legend
# ax1 = plt.gca()
handles, labels = ax1.get_legend_handles_labels()
colors_handles = handles[::3]  # 假设有三种颜色
colors_labels = ['DNA-to-DNA', 'Image-to-Image', 'Image-to-DNA']
colors_legend = ax1.legend(colors_handles, colors_labels, loc='lower left', bbox_to_anchor=(0.2, 0), fontsize=16)
colors_legend_right = ax1.legend(colors_handles, colors_labels, title='Color', loc='lower left', bbox_to_anchor=(1.4, 0))

ax1.clear()
sns.lineplot(data=macro_df, x='Taxonomy Level', y='Accuracy', hue='Model', palette=['grey'], style='Model', markers=False, dashes=linestyle_group, ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
marker_handles = handles[0:3]  # 假设有三种颜色
marker_labels = ['Seen', 'Unseen', 'H.M.']
marker_legend = ax1.legend(marker_handles, marker_labels, loc='lower left', bbox_to_anchor=(0, 0), fontsize=16)
marker_legend_right = ax1.legend(marker_handles, marker_labels, title='Linestyle', loc='lower left', bbox_to_anchor=(1.2, 0))
ax1.clear()

sns.lineplot(data=macro_df, x='Taxonomy Level', y='Accuracy', hue='Model', palette=color_group, style='Model', markers='.', dashes=linestyle_group, ax=ax1)
ax1.get_legend().remove()
fig.add_artist(colors_legend)
fig.add_artist(marker_legend)
ax1.set_ylim(ymin=0)

for model in macro_df['Model'].unique():
    model_data = macro_df[macro_df['Model'] == model]
    ax1.fill_between(model_data['Taxonomy Level'], model_data['Lower'], model_data['Upper'], alpha=0.3, color=filler_group[model])


sns.lineplot(data=micro_df, x='Taxonomy Level', y='Accuracy', hue='Model', palette=color_group, style='Model', markers='.', dashes=linestyle_group, ax=ax2)
ax2.get_legend().remove()
# fig.add_artist(colors_legend_right)
# fig.add_artist(marker_legend_right)
ax2.set_ylim(ymin=0)

for model in micro_df['Model'].unique():
    model_data = micro_df[micro_df['Model'] == model]
    ax2.fill_between(model_data['Taxonomy Level'], model_data['Lower'], model_data['Upper'], alpha=0.3, color=filler_group[model])

# fig.suptitle('Performance (mean and std) across multiple runs for our full I+D+T (image, DNA, text) model', fontsize=18)
ax1.set_ylabel('Macro Accuracy', fontsize=20)
ax1.set_xlabel('')
ax2.set_ylabel('Micro Accuracy', fontsize=20)
ax2.set_xlabel('')
ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('')
# plt.show()
fig.tight_layout()
plt.savefig('logs/test.png')
