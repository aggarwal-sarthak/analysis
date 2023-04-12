import pandas as pd
import numpy as np

df = pd.read_csv('File.csv')
data = pd.DataFrame(df)
corr = data.corr()
clean = corr.dropna(how='all',axis=0)
clean = clean.dropna(how='all',axis=1)

upper = clean.where(np.triu(np.ones(clean.shape), k=1).astype(np.bool_))
strong = [column for column in upper.columns if any(upper[column] > 0.25)]
newdf = clean.drop(strong, axis=1)

print(newdf.columns.tolist())