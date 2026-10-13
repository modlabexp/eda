#Run this before experiment 1 to 4

!pip install pandas numpy matplotlib seaborn scikit-learn plotly -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
df = pd.read_csv('StudentsPerformance.csv')
sns.set_style('whitegrid')

#Exp 4:
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# 1. Encode categorical columns
encoded = df.copy()
cats = ['gender', 'race/ethnicity',
        'parental level of education', 'lunch',
        'test preparation course']

for col in cats:
    encoded[col] = LabelEncoder().fit_transform(encoded[col])

# 2. Pairplot grouped by test preparation
sns.pairplot(df, vars=['math score', 'reading score',
                       'writing score'],
             hue='test preparation course',
             corner=True, diag_kind='kde')
plt.show()

# 3. Interactive 3D scatter plot
fig = px.scatter_3d(
    df, x='math score', y='reading score', z='writing score',
    color='parental level of education',
    symbol='test preparation course',
    hover_data=['gender', 'lunch'],
    title='3D Student Performance'
)
fig.show()

# 4. Correlation heatmap of encoded data
corr = encoded.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, mask=np.triu(
                np.ones(corr.shape, dtype=bool)))
plt.title('Multivariate Correlation')
plt.show()

# 5. Parallel coordinates plot
fig = px.parallel_coordinates(
    encoded,
    dimensions=['math score', 'reading score', 'writing score',
                'gender', 'test preparation course'],
    color='math score',
    title='Parallel Coordinates of Student Performance'
)
fig.show()
# 6. Scatter matrix
fig = px.scatter_matrix(
    df,
    dimensions=['math score', 'reading score', 'writing score'],
    color='gender',
    symbol='test preparation course',
    hover_data=['parental level of education'],
    title='Student Score Scatter Matrix'
)
fig.update_traces(diagonal_visible=False)
fig.show()
