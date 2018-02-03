# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import Python packages

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans
#import scipy.sparse as sparse
#from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
#from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

# import PC9 shipment data into pandas df

planning_group_list = ['JCP']
avgWithinSS = {}
for name in planning_group_list:
    PC9_Shipment_Qty = pd.read_csv('C:/Users/wan616/Python_Projects/' + name + '_PC9_SHIPMENTS.csv')

# prepares data type for k-means clustering

# Kevin's Comments:
#   Need better naming conventions for variables.  Captializing a variable does not equate to a different name
#   even if the syntax is correct.  Code readability has dropped.

    x = PC9_Shipment_Qty.PC9_Shipped_Qty
    X = []
    for i in range(len(x)):
        X.append([x[i], 0]) # changes to 2D list
    X = np.asarray(X) # changes list into array

# Elbow test to determine k
# Run K-Means algorithm for all values between 1 to 10
    K = range(1,10)
    KM = [kmeans(X,k) for k in K]

# Determine the distance between each PC9 Size combination and all calculated Centroids
    centroids = [cent for (cent,var) in KM]
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]

# As all possible combinations are produced between PC9 Size and Centroids
# Keep only the pairing with the shortest distance (or MINIMUM)
    dist = [np.min(D,axis=1) for D in D_k]

# Stores all of the respective error results from each K cluster.
# As 10 clusters were run, 10 cluster results were stored
    avgWithinSS[name] = [sum(d)/X.shape[0] for d in dist]

# Initialize variables
    k = 2
    ratio = 1
    ratio2 = 1

# Perform "Elbow" test to determine the best cluster
# For each K, compare the difference in error between current K and K-1 vs 
# K and K+1 to determine where the most significant improvement in error rates are
    for i in range(1, len(avgWithinSS[name]) - 1):
        if ratio2 > ratio:
            k = i
            ratio = ratio2
        diff = avgWithinSS[name][i - 1] - avgWithinSS[name][i]
        diff2 = avgWithinSS[name][i] - avgWithinSS[name][i + 1]
        ratio2 = diff - diff2

# k-means clustering by PC9 volume
# Re-Run K-Means clustering algorithm for the specific K value as determined by the Elbow Test
    list_k = [i for i in range(k)]

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

# Plot the results of the K-Means algorithm
    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], list_k, 
        markers='^', markeredgewidth=2)

# Store the results of the algorithm back within the original data set
    PC9_Shipment_Qty['PC9_Vol_Cluster_Unsorted'] = kmeans.labels_ 

# Order Volume Clusters by Avg. Shipment Vol to order cluster from Smallest to Largest
    Volume_Cluster_Definitions = PC9_Shipment_Qty.groupby(['PC9','PC9_Vol_Cluster_Unsorted']).sum().groupby('PC9_Vol_Cluster_Unsorted').mean()
    Volume_Cluster_Definitions = Volume_Cluster_Definitions.sort_values(by=['PC9_Shipped_Qty'])
    Volume_Cluster_Definitions['Unsorted_Cluster'] = Volume_Cluster_Definitions.index.get_values()
    Sorted_Grouping_List = [i for i in range(len(Volume_Cluster_Definitions.index))]
    Volume_Cluster_Definitions['Sorted_Cluster'] = Sorted_Grouping_List
    
# Re-apply new Volume Cluster Definitions to original dataframe
    PC9_Shipment_Qty = PC9_Shipment_Qty.merge(Volume_Cluster_Definitions, left_on='PC9_Vol_Cluster_Unsorted', right_on='Unsorted_Cluster', how='inner')
    
# Create a new DataFrame of only PC9 and Respective Volume Clusters
    PC9_Vol_Clusters = PC9_Shipment_Qty[['PC9', 'Sorted_Cluster']].copy()
        
# Separate out all of the PC9s for each respective Volume Cluster into a separate Numpy Array object
# This enables easier processing, optimization, and modification of each cluster
    PC9_Shipment_Qty_array = PC9_Shipment_Qty.values # change to np array
    n = len(PC9_Shipment_Qty_array[0]) - 1 # cluster index of element in np array
    m = PC9_Shipment_Qty['Sorted_Cluster'].max() # max cluster number
    filtered_all = []
    
    for i in range(m + 1):
        filtered = []
        
        for x in PC9_Shipment_Qty_array:
            if x[n] == i:
                filtered.append(x)
        filtered_all.append(filtered)

# Assign PC9 volume clusters to unique dataframes

    cluster_categories = list_k
    cluster_df = {}

    for cluster in cluster_categories:
        df_name = 'PC9_volume_cluster_' + str(cluster)
        file = filtered_all[cluster]
        cluster_df[df_name] = pd.DataFrame(file, columns = ['Planning_Group', 'Fiscal_Year', 'Season', 'Consumer_Group', 'PC9', 'PC9_Shipped_QTY', 'PC9_Vol_Cluster_Unsorted1','PC9_Avg_Ship_Qty','Unsorted_Cluster','Sorted_Cluster'])

    # import size shipment data with PC9 grouping and reformat

    PC9_size_shipment_QTY = pd.read_csv(name + '_PC9_SIZE_SHIPMENTS.csv')
    PC9_size_shipment_QTY.columns = ['PC9', 'Size_1', 'Size_2', 'Size_Shipped_Qty']
    
    # join size data with PC9 shipment data

    joined_data = {}
    temp_data = {}

    for cluster in cluster_categories:
        df_name = 'PC9_volume_cluster_' + str(cluster)
        joined_data[df_name] = pd.merge(cluster_df[df_name], PC9_size_shipment_QTY, on=['PC9'])
        temp_data[df_name] = joined_data[df_name].drop(['Consumer_Group', 'PC9_Avg_Ship_Qty', 'Sorted_Cluster'], axis = 1)

    # group size shipment data by unique sizes and then sort Size_Shipped_QTY ascending

    grouped_data = {}

    for cluster in cluster_categories:
        df_name = 'PC9_volume_cluster_' + str(cluster)
        grouped_data[df_name] = pd.DataFrame(temp_data[df_name].groupby(['Size_1', 'Size_2']).sum().sort_values(by = ['Size_Shipped_Qty']).reset_index())

    # collects sum of Size_Shipped_QTY sorted and row count for each cluster

    sum_shipped_qty = {}
    row_count = {}
    cluster_sum = []
    percent_retention = {}

    for cluster in cluster_categories:
        df_name = 'PC9_volume_cluster_' + str(cluster)
        cluster_sum.append(grouped_data[df_name].Size_Shipped_Qty.sum())

    cluster_sum = sorted(cluster_sum)

    for cluster in cluster_categories:
        df_name = 'PC9_volume_cluster_' + str(cluster)
        number_rows = len(grouped_data[df_name])
        sum_shipped_qty[df_name] = cluster_sum[cluster]
        row_count[df_name] = number_rows

    # optimization process (decided to do a 98% retention on all clusters for simplicity and consistency)

    for i in range(len(cluster_categories)):
        df_name = 'PC9_volume_cluster_' + str(i)
        percent_retention[df_name] = 0.99

    for cluster in cluster_categories:
        cum_sum = 0
        counter = 0
        df_name = 'PC9_volume_cluster_' + str(cluster)
        
        for i in range(row_count[df_name]):
            cum_sum = cum_sum + grouped_data[df_name].Size_Shipped_Qty[i]
            if cum_sum < (1 - percent_retention[df_name]) * sum_shipped_qty[df_name]:
                counter = counter + 1
            else:
                break
        
        grouped_data[df_name] = grouped_data[df_name].drop(grouped_data[df_name].index[0:counter])
   
    # enforce high volume size grid count constraint
    
#    silly_business_constraint_clusters = []
#    max_cluster = max(cluster_categories)
    
#    for i in range(len(cluster_categories) - 1):
#        df_name = 'PC9_volume_cluster_' + str(i)
#        size_count_diff = len(grouped_data[df_name]) - len(grouped_data['PC9_volume_cluster_' + str(max_cluster)])
#        if size_count_diff <= 0:
#            silly_business_constraint_clusters.append(i)
    
#    silly_business_constraint_clusters.append(max_cluster)
    
    # join size volume cluster to joined_data df

    final_data = {}

    for cluster in cluster_categories:
        df_name = 'PC9_volume_cluster_' + str(cluster)
        grouped_data[df_name] = grouped_data[df_name].drop('PC9_Shipped_QTY', axis = 1)
        final_data[df_name] = pd.merge(joined_data[df_name], grouped_data[df_name], on = ['Size_1', 'Size_2'])
        final_data[df_name] = final_data[df_name].drop('PC9_Shipped_QTY', axis = 1)
        final_data[df_name].to_csv(name + ' ' + df_name + '.csv')