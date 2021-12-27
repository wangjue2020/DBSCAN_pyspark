import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs

data,target=make_blobs(n_samples=1500,n_features=2,centers=3)

# 在2D图中绘制样本，每个样本颜色不同
plt.scatter(data[:,0],data[:,1],c=target)
plt.savefig("1500.png")
df = pd.DataFrame(data)
df.to_csv("1500.txt",index=False,header=False,sep='\t')

