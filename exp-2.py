#Run this before experiment 1 to 4
!pip install pandas numpy matplotlib seaborn scikit-learn plotly -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


df = pd.read_csv('StudentsPerformance.csv')

sns.set_style('whitegrid')

#Exp 2:

num = ['math score', 'reading score', 'writing score']
cat = ['gender', 'race/ethnicity','parental level of education', 'lunch',
       'test preparation course']

# 1. Histograms with KDE
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num):
    sns.histplot(df[col], kde=True, bins=30, ax=ax[i])
    ax[i].set_title(col + ' Distribution')
plt.tight_layout()
plt.show()

# 2. Categorical count plots
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
for i, col in enumerate(cat):
    sns.countplot(data=df, y=col, ax=ax.flat[i])
    ax.flat[i].set_title(col + ' Count')
plt.tight_layout()
plt.show()

# 3. Statistical summaries
print(df[num].describe())

for col in cat:
    print('\n', col)
    print(df[col].value_counts(normalize=True) * 100)

# 4. Boxplots
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num):
    sns.boxplot(y=df[col], ax=ax[i])
    ax[i].set_title(col + ' Boxplot')
plt.tight_layout()
plt.show()

# 5. Pie charts
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
for i, col in enumerate(cat):
    df[col].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax.flat[i])
    ax.flat[i].set_title(col)
    ax.flat[i].set_ylabel('')
plt.tight_layout()
plt.show()

# 6. ECDF / Cumulative distribution
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num):
    sns.ecdfplot(data=df, x=col, ax=ax[i])
    ax[i].set_title(col + ' ECDF')
plt.tight_layout()
plt.show()

# 7. Violin plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num):
    sns.violinplot(y=df[col], ax=ax[i])
    ax[i].set_title(col + ' Violin Plot')
plt.tight_layout()
plt.show()
