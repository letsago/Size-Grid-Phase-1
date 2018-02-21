# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import Python packages

import os
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.cluster import KMeans
#import scipy.sparse as sparse
#from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
#from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

# import PC9 shipment data into pandas df

planning_group_list = ['JCP']
style_list = ['511', '505', '501', '502', '510', '512', '513', '514', '517', '527', '541', '550', '559', '569']
avgWithinSS = {}
final_data = pd.DataFrame()
for name in planning_group_list:
    for style in style_list:
        PC9ShipFile = os.path.normpath('//sfonetapp3220a/Global_MPIM/Global_Reporting_and_Analytics/Advanced_Analytics/Size_Grid_Model_Predictions/Test_Data/' + name + '_' + style + '_PC9_SHIPMENTS.csv') 
        PC9_Shipment_Qty = pd.read_csv(PC9ShipFile)
        PC9_Shipment_Qty['PC9_Shipped_Qty'] = PC9_Shipment_Qty['PC9_Shipped_Qty'].apply(str).apply(lambda x: x.replace(',','')).apply(pd.to_numeric)

    # prepares data type for k-means clustering
    
    # Kevin's Comments:
    #   Need better naming conventions for variables.  Captializing a variable does not equate to a different name
    #   even if the syntax is correct.  Code readability has dropped.
    
        x = PC9_Shipment_Qty.PC9_Shipped_Qty
        X = []
        for i in range(len(x)):
            X.append([float(x[i]), 0]) # changes to 2D list
        X = np.asarray(X) # changes list into array
    
    # Elbow test to determine k
    # Run K-Means algorithm for all values between 1 to 10
        from scipy.cluster.vq import kmeans
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
        Volume_Cluster_Definitions['PC9_Volume_Cluster'] = Sorted_Grouping_List
        
    # Re-apply new Volume Cluster Definitions to original dataframe
        PC9_Shipment_Qty = PC9_Shipment_Qty.merge(Volume_Cluster_Definitions, left_on='PC9_Vol_Cluster_Unsorted', right_on='Unsorted_Cluster', how='inner')
        
    # Create a new DataFrame of only PC9 and Respective Volume Clusters
        PC9_Vol_Clusters = PC9_Shipment_Qty[['PC9', 'PC9_Volume_Cluster']].copy()
            
    # Separate out all of the PC9s for each respective Volume Cluster into a separate Numpy Array object
    # This enables easier processing, optimization, and modification of each cluster
        PC9_Shipment_Qty_array = PC9_Shipment_Qty.values # change to np array
        n = len(PC9_Shipment_Qty_array[0]) - 1 # cluster index of element in np array
        m = PC9_Shipment_Qty['PC9_Volume_Cluster'].max() # max cluster number
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
            cluster_df[df_name] = pd.DataFrame(file, columns = ['Planning_Group', 'Fiscal_Year', 'Season', 'Consumer_Group', 'Style_Name', 'PC9', 'PC9_Shipped_QTY', 'PC9_Vol_Cluster_Unsorted','PC9_Avg_Ship_Qty','Unsorted_Cluster','PC9_Volume_Cluster'])
    
        # import size shipment data with PC9 grouping and reformat
        SizeShipFile = os.path.normpath('//sfonetapp3220a/Global_MPIM/Global_Reporting_and_Analytics/Advanced_Analytics/Size_Grid_Model_Predictions/Test_Data/' + name + '_' + style + '_PC9_SIZE_SHIPMENTS.csv')
        PC9_size_shipment_QTY = pd.read_csv(SizeShipFile)
        PC9_size_shipment_QTY.columns = ['PC9', 'Size_1', 'Size_2', 'Size_Shipped_Qty']
        PC9_size_shipment_QTY['Size_Shipped_Qty'] = PC9_size_shipment_QTY['Size_Shipped_Qty'].apply(str).apply(lambda x: x.replace(',','')).apply(pd.to_numeric)
        
        # join size data with PC9 shipment data
    
        joined_data = {}
        temp_data = {}
    
        for cluster in cluster_categories:
            df_name = 'PC9_volume_cluster_' + str(cluster)
            joined_data[df_name] = pd.merge(cluster_df[df_name], PC9_size_shipment_QTY, on=['PC9'])
            joined_data[df_name] = joined_data[df_name].drop(['PC9_Avg_Ship_Qty', 'PC9_Vol_Cluster_Unsorted', 'Unsorted_Cluster', 'PC9_Shipped_QTY'], axis = 1)
            temp_data[df_name] = joined_data[df_name].drop(['Consumer_Group', 'Style_Name', 'PC9_Volume_Cluster'], axis = 1)
    
        # group size shipment data by unique sizes and then sort Size_Shipped_QTY ascending
    
        grouped_data = {}
    
        for cluster in cluster_categories:
            df_name = 'PC9_volume_cluster_' + str(cluster)
            grouped_data[df_name] = pd.DataFrame(temp_data[df_name].groupby(['Size_1', 'Size_2']).sum().sort_values(by = ['Size_Shipped_Qty']).reset_index())
    
        # optimization process (decided to do a 99.9% retention on max volume cluster and 99.5% retention on cumulative sum on lower volume clusters)
    
        percent_retention ={}
        number_rows = {}
        cluster_sum = {}
        max_cluster = max(cluster_categories)
        
        for cluster in cluster_categories:
            counter = 0
            cum_sum = 0
            df_name = 'PC9_volume_cluster_' + str(cluster)
            cluster_sum[df_name] = grouped_data[df_name].Size_Shipped_Qty.sum()
            number_rows[df_name] = len(grouped_data[df_name])
            if cluster == max_cluster:
                percent_retention[df_name] = 0.997
                for i in range(number_rows[df_name]):
                    cum_sum = cum_sum + grouped_data[df_name].Size_Shipped_Qty[i]
                    if cum_sum < (1 - percent_retention[df_name]) * cluster_sum[df_name]:
                        counter = counter + 1
                    else:
                        break
            else:
                percent_retention[df_name] = 0.995
                for i in range(number_rows[df_name]):
                    cum_sum = cum_sum + grouped_data[df_name].Size_Shipped_Qty[i]
                    if cum_sum < (1 - percent_retention[df_name]) * cluster_sum[df_name]:
                        counter = counter + 1
                    else:
                        break
            grouped_data[df_name] = grouped_data[df_name].drop(grouped_data[df_name].index[0:counter]) 
                    
        # higher volume size grid count constraint by checking if lower volume size grid is subset
        
        constraint_data = {}
        temp_data_2 = {}
        
        for cluster in cluster_categories:
            df_name = 'PC9_volume_cluster_' + str(cluster)
            temp_data_2[df_name] = grouped_data[df_name]
        
        for i in list(reversed(range(len(cluster_categories)))):
            df_name_1 = 'PC9_volume_cluster_' + str(i)
            if i == max(cluster_categories):
                constraint_data[df_name_1] = grouped_data[df_name_1]
            if i > 0:
                df_name_2 = 'PC9_volume_cluster_' + str(i - 1)
                temp_data_2[df_name_2] = pd.merge(temp_data_2[df_name_1], temp_data_2[df_name_2], on = ['Size_1', 'Size_2'])
                temp_data_2[df_name_2] = temp_data_2[df_name_2].drop('Size_Shipped_Qty_x', axis = 1)
                temp_data_2[df_name_2].columns = ['Size_1', 'Size_2', 'Size_Shipped_Qty']
                constraint_data[df_name_2] = temp_data_2[df_name_2]    
        
        # export one output file
        
        for cluster in cluster_categories:
            df_name = 'PC9_volume_cluster_' + str(cluster)
            constraint_data[df_name] = constraint_data[df_name].drop('Size_Shipped_Qty', axis = 1)
            final_data = final_data.append(pd.merge(joined_data[df_name], constraint_data[df_name], on = ['Size_1', 'Size_2']))
final_data.to_csv('//sfonetapp3220a/Global_MPIM/Global_Reporting_and_Analytics/Advanced_Analytics/Size_Grid_Model_Predictions/Raw_Data/' + 'MASTER' + '.csv')