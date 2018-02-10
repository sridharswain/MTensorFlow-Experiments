import tensorflow as tf
import numpy as np
def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float64, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples

def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    import matplotlib.pyplot as plt
    #Plot out the different clusters
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
         samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
         plt.scatter(samples[:,0], samples[:,1], c=colour[i])
         plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=10)
         plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='m', mew=5)
    plt.show()
