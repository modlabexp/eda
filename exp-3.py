#Run this before experiment 1 to 4

!pip install pandas numpy matplotlib seaborn scikit-learn plotly -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
df = pd.read_csv('StudentsPerformance.csv')
sns.set_style('whitegrid')

#Exp 3:

num = ['math score', 'reading score', 'writing score']
cat = ['gender', 'race/ethnicity',
       'parental level of education', 'lunch',
       'test preparation course']

# 1. Numerical vs Numerical: Pairplot
sns.pairplot(df, vars=num, diag_kind='kde', corner=True)
plt.show()

# 2. Correlation heatmap
sns.heatmap(df[num].corr(), annot=True,
            cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Score Correlation')
plt.show()

# 3. Categorical vs Numerical: Boxplots
for c in cat:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, s in enumerate(num):
        sns.boxplot(data=df, x=c, y=s, ax=ax[i])
        ax[i].tick_params(axis='x', rotation=45)
        ax[i].set_title(s + ' by ' + c)
    plt.tight_layout()
    plt.show()

# 4. Categorical vs Categorical: Cross-tab heatmaps
for a, b in [('gender', 'race/ethnicity'),
             ('gender', 'test preparation course'),
             ('lunch', 'race/ethnicity'),
             ('lunch', 'test preparation course')]:
    table = pd.crosstab(df[a], df[b], normalize='index')
    sns.heatmap(table, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title(a + ' vs ' + b)
    plt.show()

# 5. Violin plots: Test preparation + Gender
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, s in enumerate(num):
    sns.violinplot(data=df, x='test preparation course',
                   y=s, hue='gender', split=True, ax=ax[i])
    ax[i].set_title(s + ' by Test Prep and Gender')
plt.tight_layout()
plt.show()
