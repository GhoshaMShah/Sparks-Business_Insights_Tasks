import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()
irisDf = pd.DataFrame(iris.data, columns = iris.feature_names)

X = irisDf.iloc[:,:-1].values
#finding optimal number of clusters using the elbow method  
from sklearn.cluster import KMeans  
wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)  
    wcss_list.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss_list)  
plt.title('The Elobw Method')  
plt.xlabel('Number of clusters')  
plt.ylabel('wcss_list')
plt.show()  

#training the K-means model on a dataset  
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_predict= kmeans.fit_predict(X)  

#visulaizing the clusters  
plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], s = 100, c = 'blue', label = 'Iris-setosa') #for first cluster  
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], s = 100, c = 'green', label = 'Iris-versicolour') #for second cluster  
plt.scatter(X[y_predict== 2, 0], X[y_predict == 2, 1], s = 100, c = 'red', label = 'Iris-virginica') #for third cluster  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.title('Clusters for Iris')  
plt.show()  