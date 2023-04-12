import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('File.csv')
data = pd.DataFrame(df)
corr = data.corr()
clean = corr.dropna(how='all',axis=0)
clean = clean.dropna(how='all',axis=1)

upper = clean.where(np.triu(np.ones(clean.shape), k=1).astype(np.bool_))
strong = [column for column in upper.columns if any(upper[column] > 0.75)]
newdf = clean.drop(strong, axis=1)

plt.figure(figsize=(16,16))
heat_map = sns.heatmap(newdf, linewidth = 1)
plt.show()