# Clustering Cardiac Arrest Data

This project involves performing clustering analysis on a cardiac arrest dataset to identify patterns and group similar cases. The analysis includes preprocessing steps, applying both hierarchical and K-means clustering techniques, and evaluating their performance.

## Key Features:

1. **Imported File**: Loaded the cardiac arrest dataset for analysis.
2. **Checked for Null Values**: Identified and handled missing values in the dataset.
3. **Pair Plot Visualization**: Created a pair plot between the 'radius_mean' and 'texture_mean' columns, segmented by diagnosis.
4. **Feature Selection**: Selected 'radius_mean' and 'texture_mean' as features for clustering and created a new dataset.
5. **Data Scaling**: Applied standard scaling to the new dataset to normalize feature values.
6. **Hierarchical Clustering**: Displayed a dendrogram using Scipy to visualize hierarchical clustering.
7. **Agglomerative Clustering**: Applied Agglomerative Clustering with 2 clusters, predicted the cluster labels, and created a column for these labels.
8. **Label Analysis**: Checked and plotted the count of each label.
9. **Silhouette Score**: Evaluated clustering quality with the silhouette score.
10. **K-Means Clustering**: Applied K-means clustering with 2 clusters, checked the WCSS score (94%), and experimented with different numbers of clusters (1 to 10) to find the optimal number.
11. **Final K-Means Clustering**: Applied K-means with the best number of clusters based on the WCSS score and created a column for cluster labels.

## Why This Project?

This project demonstrates a comprehensive approach to clustering analysis, including data preprocessing, feature selection, and applying both hierarchical and K-means clustering techniques. By comparing different clustering methods and evaluating their performance, the project provides insights into how to effectively segment data and interpret clustering results.

## What You Will Learn:

- Handling and preprocessing data for clustering tasks.
- Visualizing data relationships using pair plots.
- Applying and evaluating hierarchical and K-means clustering techniques.
- Selecting the optimal number of clusters based on WCSS and silhouette scores.
- Practical insights into clustering real-world datasets for pattern recognition.

## Conclusion

The analysis of the cardiac arrest dataset using hierarchical and K-means clustering techniques revealed valuable insights into data segmentation. Hierarchical clustering provided a dendrogram for visualizing cluster relationships, while K-means clustering, with its evaluation through WCSS and silhouette scores, demonstrated effective cluster identification. This project highlights the importance of preprocessing, feature selection, and methodical evaluation in clustering analysis.
