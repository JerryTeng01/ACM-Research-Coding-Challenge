import pandas as pd
from sklearn.cluster import DBSCAN

data = pd.read_csv("./ClusterPlot.csv", sep=',', usecols=['V1', 'V2'])
model = DBSCAN(eps=.25, min_samples=10)
clusters = model.fit_predict(data)
numClusters = len(set(clusters)) - (-1 in clusters)
print("Predicted", numClusters, "clusters")
