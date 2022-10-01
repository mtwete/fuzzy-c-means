"""
Matthew Twete

Implementation of the fuzzy c-means algorithm on a cluster dataset.
"""

#Import needed libraries
import numpy as np
import random
import matplotlib.pyplot as plt

#Import the data
data = np.genfromtxt('cluster_dataset.txt')

#Fuzzy c-means algorithm class, allowing you to pass in clustering data and set the value
#of k (how many clusters), m (the fuzzifier value) and how many runs you want the algorithm to do. 
class cmeans:
    #Fuzzy c-means algorithm class constructor. The arguments are:
    #data is the data to be clustered
    #k is the number of clusters to separate the data into
    #r is the number of times to run the algorithm 
    #m is the fuzzifier value to run with
    def __init__(self, Data, k, r, m):
        #Clustering data
        self.data = Data
        #Number of clusters
        self.k = k
        #Number of runs
        self.r = r
        #Data structure to hold the cluster centroids for each run
        self.centroids = np.ones((r,k,data.shape[1]))
        #Data structure to hold the cluster centroids at every iteration for each run
        self.prevcentroids = [[] for _ in range(self.r)]
        #Data structure to hold the cluster weights at every iteration for each run
        self.prevweights = [[] for _ in range(self.r)]
        #Array to hold the final sum squared error for each run
        self.final_error = np.zeros(r)
        #Data structure to hold the data points in each cluster for plotting
        self.clusters = [[] for _ in range(self.k)]   
        #The fuzzifier value to run with
        self.m = m
        #Data structure to hold the cluster membership weights of each data point for each run
        self.weights = np.random.uniform(0,1,(r,data.shape[0],k))
    
    
    #Function to calculate the cluster membership weight updates.
    #The only arguments is:
    #run, an integer representing which run the algorithm is on, zero indexed
    def calc_weights(self,run):
        #Loop over the data
        for i in range(self.data.shape[0]):
            #Loop over each cluster
            for j in range(self.k):
                #Variable to hold the value of the sum in the weight update formula
                total = 0
                #Variable to hold the numerator value inside the sum of the weight update formula
                numerator = np.linalg.norm(self.data[i]-self.centroids[run][j])
                #Loop over the clusters again, to calculate the denominator value inside the sum of the weight update formula
                for k in range(self.k):
                    #Add the numerator divided by the denominator for each value in the sum
                    total += (numerator/np.linalg.norm(self.data[i]-self.centroids[run][k]))**(2/(self.m-1))
                #Once the total sum is calculated, update the cluster membership value of that data point for the cluster
                self.weights[run][i][j] = 1/total
          
                
    #Function to re-calculate the cluster centroids based on data points cluster membership grades.          
    #The only arguments is:
    #run, an integer representing which run the algorithm is on, zero indexed
    def calc_centroids(self,run):
        #Loop over each cluster and calculate its centroid based on the data point cluster membership grades
        for i in range(self.k):
            #Update the centroid of the cluster
            self.centroids[run][i] = np.dot(np.transpose(self.weights[run][:,i]**self.m),self.data)/sum(self.weights[run][:,i]**self.m)
                
    
    #Function to calculate the sum squared error for given centroid values, the function will
    #return the sum squared error of the data for the given cluster centroids. The only arguments is:
    #means, array containing the centroids of each cluster
    #weight, array contraining the membership weights of the data                                                                 
    def sum_sq_er(self,means,weight):
        #Variable to hold the sum squared error
        sse = 0
        #Get the cluster of each data point
        closest = weight.argmax(axis=1)
        #Loop over the clusters
        for l in range(self.k):
            #For the current cluster, get the data points in the cluster
            currentClust = np.where(closest==l,1,0)
            #Subtract the positions of the centroid from each data point
            delta = data-means[l][:]
            #Pick out only the data points in that cluster
            delta = currentClust.reshape(-1,1)*delta
            #Add the L2 squared distance of each of the data points in the cluster
            for i in range(self.data.shape[0]):
                sse += np.linalg.norm(delta[i])**2
        return sse  
    
    
    #Function to separate the data into the clusters that they have the highest weight membership of.
    #The only arguments is:
    #weight, array contraining the membership weights of the data  
    def partition_clusters(self,weight):
        #Get the closest cluster of each data point
        closest = weight.argmax(axis=1)
        #Loop over the clusters
        for l in range(self.k):
            #For the current cluster, get the data points in the cluster
            currentClust = np.where(closest==l,1,0)
            #Set all data points not in the cluster to 0
            clusterPoints = currentClust.reshape(-1,1)*data
            #Pick out only the data points in the cluster (ie the ones not equal to 0)
            clusterPoints = clusterPoints[~np.all(clusterPoints == 0,axis=1)]
            #Store those data points
            self.clusters[l] = clusterPoints
    
    
    #Function to calculate the sum squared error for each run of the algorithm.
    def calc_errors(self):
        #Loop over the final centroids of each run and calculate the sum squared error
        for i in range(self.r):
            self.final_error[i] = self.sum_sq_er(self.centroids[i],self.weights[i])
            
            
    #Function to plot the results of the algorithm. It will plot the cluster centroids and data points
    #in those clusters from the best run of the algorithm. It will plot the centroids and clusters from the fifth iteration,
    #the centroids and clusters from the iteration in the middle of the run and then the final centroids and 
    #clusters.
    def plot_results(self):
        #Set up plots
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #Get the index of the best run of the algorithm
        bestrun = np.argmin(self.final_error)
        #Get the centroids from iteration 5 from the best run of the algorithm, and separate the data into the clusters
        self.best_centroid = self.prevcentroids[bestrun][4]
        self.partition_clusters(self.prevweights[bestrun][4])
        #Loop over each cluster (except the last) and plot the centroid and data points in that cluster.
        #This is done so that each cluster's data points have a different color. 
        for i in range(self.k-1):
            ax1.scatter(self.clusters[i][:,0],self.clusters[i][:,1], s = np.full(len(self.clusters[i]),3),label = "Cluster " + str(i+1) + " data")
            ax1.scatter(self.best_centroid[i][0],self.best_centroid[i][1],marker='v',c=1)
        #Plot the centroid and data of the last cluster, the reason this is done separately is simply to make the 
        #plot legend formatted more neatly, since centroids are all plotted as the same shape and color and I want the 
        #legend to only have one row for the centroid points
        ax1.scatter(self.clusters[self.k-1][:,0],self.clusters[self.k-1][:,1], s = np.full(len(self.clusters[self.k-1]),3),label = "Cluster " + str(self.k) + " data")
        ax1.scatter(self.best_centroid[self.k-1][0],self.best_centroid[self.k-1][1],marker='v',c=1,label = "Cluster centroids")
        #Add a legend, title, axis labels and show the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('X-value of data')
        plt.ylabel('Y-value of data')
        plt.title("Plot of best fuzzy c-means model with " + str(self.k) + " clusters and m set to "+ str(self.m) + " on iteration 5")
        plt.show()
        
        #Set up plots
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #Get the centroids from the middle most iteration of the best run of the algorithm, and separate the data into the clusters
        self.best_centroid = self.prevcentroids[bestrun][int(len(self.prevcentroids[bestrun])/2)]
        self.partition_clusters(self.prevweights[bestrun][int(len(self.prevweights[bestrun])/2)])
        #Loop over each cluster (except the last) and plot the centroid and data points in that cluster.
        #This is done so that each cluster's data points have a different color. 
        for i in range(self.k-1):
            ax1.scatter(self.clusters[i][:,0],self.clusters[i][:,1], s = np.full(len(self.clusters[i]),3),label = "Cluster " + str(i+1) + " data")
            ax1.scatter(self.best_centroid[i][0],self.best_centroid[i][1],marker='v',c=1)
        #Plot the centroid and data of the last cluster, the reason this is done separately is simply to make the 
        #plot legend formatted more neatly, since centroids are all plotted as the same shape and color and I want the 
        #legend to only have one row for the centroid points
        ax1.scatter(self.clusters[self.k-1][:,0],self.clusters[self.k-1][:,1], s = np.full(len(self.clusters[self.k-1]),3),label = "Cluster " + str(self.k) + " data")
        ax1.scatter(self.best_centroid[self.k-1][0],self.best_centroid[self.k-1][1],marker='v',c=1,label = "Cluster centroids")
        #Add a legend, title, axis labels and show the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('X-value of data')
        plt.ylabel('Y-value of data')
        plt.title("Plot of best fuzzy c-means model with " + str(self.k) + " clusters and m set to "+ str(self.m) + " on iteration " + str(int(len(self.prevcentroids[bestrun])/2)))
        plt.show()
        
        #Set up plots
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #Get the final centroids from the best run of the algorithm, and separate the data into the clusters
        self.best_centroid = self.centroids[bestrun]
        self.partition_clusters(self.weights[bestrun])
        #Loop over each cluster (except the last) and plot the centroid and data points in that cluster.
        #This is done so that each cluster's data points have a different color. 
        for i in range(self.k-1):
            ax1.scatter(self.clusters[i][:,0],self.clusters[i][:,1], s = np.full(len(self.clusters[i]),3),label = "Cluster " + str(i+1) + " data")
            ax1.scatter(self.best_centroid[i][0],self.best_centroid[i][1],marker='v',c=1)
        #Plot the centroid and data of the last cluster, the reason this is done separately is simply to make the 
        #plot legend formatted more neatly, since centroids are all plotted as the same shape and color and I want the 
        #legend to only have one row for the centroid points
        ax1.scatter(self.clusters[self.k-1][:,0],self.clusters[self.k-1][:,1], s = np.full(len(self.clusters[self.k-1]),3),label = "Cluster " + str(self.k) + " data")
        ax1.scatter(self.best_centroid[self.k-1][0],self.best_centroid[self.k-1][1],marker='v',c=1,label = "Cluster centroids")
        #Add a legend, title, axis labels and show the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('X-value of data')
        plt.ylabel('Y-value of data')
        plt.title("Plot of best fuzzy c-means model with " + str(self.k) + " clusters and m set to "+ str(self.m) + " clusters after final iteration")
        plt.show()
        
        
    #Function to run the algorithm, it will handle all the calcualations and display the results
    def run(self):
        #Run the algorithm r times
        for i in range(self.r):
            #Pick initial centroids at random from the data
            for j in range(self.k):
                self.centroids[i][j] = random.choice(self.data)
            #Set up data structure to hold the previous iteration's centroids, to be used in the stopping condition
            prevcent = np.zeros(self.centroids[i].shape)
            #Loop until the previous iteration centroids are essentially the same as the current centroids
            while(np.all(abs(self.centroids[i]-prevcent) > 0.00001)):
                #Save the centroids so they can be used to plot the algorithm at different iterations
                self.prevcentroids[i].append(np.copy(self.centroids[i]))
                #Save the weights so they can be used to plot the algorithm at different iterations
                self.prevweights[i].append(np.copy(self.weights[i]))
                #Save the previous iteration's centroids
                prevcent = np.copy(self.centroids[i])
                #Recalculate the centroids and weights
                self.calc_centroids(i)
                self.calc_weights(i)

        #Calculate the sum squared error for each run
        self.calc_errors()
        #Plot the results from the best run
        self.plot_results()
        #Print the sum squared error of the best run
        print("Sum Squared Error of the best fuzzy c-means model with " + str(self.k) + " clusters and m set to "+ str(self.m) + ": " + str(np.amin(self.final_error)))
                
    
            

#Run the fuzzy c-means algorithm with 3, 5 and 8 clusters for 10 runs each with m set to 3

c = cmeans(data,3,10,3)
c.run()

c = cmeans(data,5,10,3)
c.run()

c = cmeans(data,8,10,3)
c.run() 