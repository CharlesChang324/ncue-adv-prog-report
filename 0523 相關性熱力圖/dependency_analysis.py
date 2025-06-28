import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# 讀取數據
df = pd.read_csv('machine_failure_cleaned.csv')

# 1. 連續變數之間的相關性分析
continuous_vars = ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# 計算皮爾遜相關係數
pearson_corr = df[continuous_vars].corr(method='pearson')
spearman_corr = df[continuous_vars].corr(method='spearman')

# 繪製熱力圖
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation')

plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation')

plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 2. 散佈圖矩陣
# 設置更大的圖形大小
plt.figure(figsize=(15, 15))
g = sns.pairplot(df[continuous_vars + ['Machine failure']], 
                 hue='Machine failure',
                 height=4,
                 aspect=1.2,
                 plot_kws={'alpha': 0.6})

# 調整標籤
for i in range(len(continuous_vars)):
    for j in range(len(continuous_vars)):
        if i != j:  # 非對角線圖
            g.axes[i, j].set_xlabel(continuous_vars[j], fontsize=10)
            g.axes[i, j].set_ylabel(continuous_vars[i], fontsize=10)
        else:  # 對角線圖
            g.axes[i, j].set_xlabel(continuous_vars[i], fontsize=10)
            g.axes[i, j].set_ylabel('Density', fontsize=10)

# 調整圖例位置和大小
plt.legend(title='Machine Failure', 
          labels=['No Failure', 'Failure'],
          bbox_to_anchor=(1.02, 0.5),
          loc='center left')

plt.tight_layout()
plt.savefig('scatter_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. 類別變數與連續變數的關係
failure_vars = ['TWF', 'HDF', 'PWF', 'OSF', 'Machine failure']

# 計算點二系列相關係數
point_biserial_corr = {}
for var in continuous_vars:
    point_biserial_corr[var] = stats.pointbiserialr(df['Machine failure'], df[var])

# 4. 類別變數之間的關係
# 計算卡方檢定
chi2_results = {}
for i in range(len(failure_vars)):
    for j in range(i+1, len(failure_vars)):
        contingency = pd.crosstab(df[failure_vars[i]], df[failure_vars[j]])
        chi2, p, dof, expected = chi2_contingency(contingency)
        chi2_results[f'{failure_vars[i]} vs {failure_vars[j]}'] = {'chi2': chi2, 'p-value': p}

# 輸出結果
print("\n=== 連續變數相關係數 ===")
print("\nPearson相關係數:")
print(pearson_corr)
print("\nSpearman相關係數:")
print(spearman_corr)

print("\n=== 點二系列相關係數 (與機器故障) ===")
for var, result in point_biserial_corr.items():
    print(f"{var}: r = {result.correlation:.3f}, p-value = {result.pvalue:.3f}")

print("\n=== 類別變數卡方檢定 ===")
for pair, result in chi2_results.items():
    print(f"{pair}: chi2 = {result['chi2']:.3f}, p-value = {result['p-value']:.3f}")

# 5. 視覺化類別變數之間的關係
plt.figure(figsize=(10, 8))
sns.heatmap(pd.crosstab(df['Machine failure'], df['TWF']), annot=True, fmt='d', cmap='YlOrRd')
plt.title('Machine Failure vs TWF')
plt.savefig('failure_vs_twf.png')
plt.close() 