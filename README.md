# ACM Research Coding Challenge (Fall 2020)

For the challenge, I used the Scikit and Pandas libraries. To get to my solution, I watched some Youtube videos and read a couple articles for an introduction to different clustering algorithms, then I chose one and went to the documentation page for it. In this case, I chose to use the DBSCAN clustering algorithm since it seemed the most appropriate for the similarly dense clusters in the data set and it would allow me to find the number of clusters. The DBSCAN algorithm uses the concept of core, boundary, and outlier points to assign each data point a state and expands cluster groups using those states. I got most of my information from the DBSCAN documentation page (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). For the epsilon and mininum samples parameters, I chose somewhat arbitrary numbers through some trial and error that would allow me to get distinct clusters without being too loose or tight. I applied my model to the data from the csv file and to find the number of clusters, I used the set function to find distinct elements in the list of cluster labels returned from fit_predict and subtracted the existence of a noise cluster signified by the number -1.
