import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("fid_score.csv",header=None,comment="#")
x=df[0]
x=x.to_numpy()
a=df[1]
a=a.to_numpy()  
b=df[2]
b=b.to_numpy()
c=df[3]
c=c.to_numpy()
plt.plot(x,a,label="custom")
plt.plot(x,b,label="torchmetrics")
plt.plot(x,c,label="torcheval")
plt.xlabel("epochs")
plt.ylabel("fid score")
plt.legend()
