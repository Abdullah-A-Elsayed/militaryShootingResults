from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def getNewBullets (Bullets_aggregated, FinalImageBulletsNo):
    X = np.array( Bullets_aggregated, np.float64)
    scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    km = KMeans(n_clusters=5, init="k-means++", n_init=FinalImageBulletsNo,
        max_iter=300,
        random_state=42)
    km.fit(X)
    labels = list(km.labels_)
    new_bullets = []
    for index, label in enumerate(labels):
        i = labels.index(label)
        partA = labels[0:i]
        if (type(partA) is not list):
            partA = [partA]
        partB = labels[i+1:]
        if (type(partB) is not list):
            partB = [partB]
        rest = partA+partB
        j = 1 if label in rest else -1
        if j == -1:
            new_bullets.append(list(X[index]))

    return new_bullets




Bullets_aggregated = [
              [1,1],
              [5,8],
              [26,22],
              
              [3,2],
              [6,9],
              [27,23],

              [11,10],
              [15,15]
              ]
print(getNewBullets(Bullets_aggregated, 5))