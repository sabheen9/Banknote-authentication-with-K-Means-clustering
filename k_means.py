import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 

#Loading the file
data=pd.read_csv("k_means_project.csv")
#Normalizing the data
Normalized= (data - data.min())/(data.max() - data.min())

x = Normalized['V1']
y = Normalized['V2']
xy = np.column_stack((x,y))
#finding the center of the clusters
km_res = KMeans(n_clusters=2).fit(xy)
centers = km_res.cluster_centers_

#Assign each value to its cluster
km_res_fit = KMeans(n_clusters=2).fit_predict(xy)
Normalized['cluster'] = km_res_fit
print(Normalized.head())

# Visualise the data
plt.scatter(x, y, c = km_res_fit)
plt.scatter(centers[:,0], centers[:,1], s = 400)
plt.show()

print(centers)
Normalized['cluster'].value_counts()