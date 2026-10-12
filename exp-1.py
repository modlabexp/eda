#Run this before experiment 1 to 4

!pip install pandas numpy matplotlib seaborn scikit-learn plotly -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['StudentsPerformance.csv']))
sns.set_style('whitegrid')

#Exp 1:
# 1. Inspect data
print(df.head())
df.info()
print(df.isnull().sum())

# 2. Handle missing values
data = df.copy()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())

# Keep original categories for visualization
raw = data.copy()

# 3. Label encoding for binary columns
from sklearn.preprocessing import LabelEncoder, StandardScaler

for col in ['gender', 'lunch', 'test preparation course']:
    data[col] = LabelEncoder().fit_transform(data[col])

# 4. One-hot encoding for multi-category columns
data = pd.get_dummies(
    data,
    columns=['race/ethnicity', 'parental level of education'],
    drop_first=True,
    dtype=int
)
# 5. Standardize scores
scores = ['math score', 'reading score', 'writing score']
data[scores] = StandardScaler().fit_transform(data[scores])
print("\nPreprocessed data:")
print(data.head())

# 6. Score distributions
raw[scores].hist(figsize=(12, 4), bins=20)
plt.tight_layout()
plt.show()

# 7. Scores by gender
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(scores):
    sns.boxplot(data=raw, x='gender', y=col, ax=ax[i])
    ax[i].set_title(col + ' by Gender')
plt.tight_layout()
plt.show()

# 8. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm',
            center=0, mask=np.triu(np.ones(
                (len(data.columns), len(data.columns)), dtype=bool)))
plt.title('Correlation Heatmap')
plt.show()

# 9. Test preparation impact
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(scores):
    sns.barplot(data=raw, x='test preparation course',
                y=col, ax=ax[i])
    ax[i].set_title(col + ' by Test Preparation')
plt.tight_layout()
plt.show()

# 10. Pairplot
sns.pairplot(raw, vars=scores, hue='gender', corner=True)
plt.show()

# 11. Lunch type impact
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(scores):
    sns.boxplot(data=raw, x='lunch', y=col, ax=ax[i])
    ax[i].set_title(col + ' by Lunch Type')
plt.tight_layout()
plt.show()

print("Preprocessing and visualization completed.")

