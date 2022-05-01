import sys
import math
import numpy as np
import scipy.io.wavfile


def read_wav_file(wav_filename):
    rate, wav_data = scipy.io.wavfile.read(wav_filename)
    return np.array(wav_data)


def read_centroids(centroids_filename):
    centroids = np.loadtxt(centroids_filename)
    return centroids


def calc_distance(data_sample, centroid):
    d0 = data_sample[0] - centroid[0]
    d1 = data_sample[1] - centroid[1]
    return math.sqrt(d0 * d0 + d1 * d1)


def compute_closest_centroid_index(data_sample, centroids):
    distances = [calc_distance(data_sample, centroid) for centroid in centroids]
    return int(np.argmin(distances))


def compute_average_value(cluster):
    return np.average(cluster, axis=0)


def compute_new_centroids(clusters):
    centroids = np.array([compute_average_value(cluster) for cluster in clusters])
    return np.round(centroids)


# fill empty clusters with its own centroid
def fill_empty_clusters(clusters, centroids):
    for i in range(len(clusters)):
        if len(clusters[i]) == 0:
            clusters[i].append(centroids[i])


def k_means_iteration(centroids, data):
    # data clusters
    clusters = [[] for _ in range(len(centroids))]

    # assign data sample to the closest centroid cluster
    for data_sample in data:
        closest_centroid_index = compute_closest_centroid_index(data_sample, centroids)
        clusters[closest_centroid_index].append(data_sample)

    fill_empty_clusters(clusters, centroids)

    new_centroids = compute_new_centroids(clusters)
    return new_centroids


def did_converge(old_centroids, centroids):
    return np.array_equal(old_centroids, centroids)


def main(wav_filename, centroids_filename):
    wav_data = read_wav_file(wav_filename)
    centroids = read_centroids(centroids_filename)

    logger = Logger('output.txt')

    n_epochs = 30
    for epoch in range(n_epochs):
        old_centroids = centroids.copy()  # save copy of previous centroids
        centroids = k_means_iteration(centroids, wav_data)  # update the centroids
        logger.log_iteration_data(epoch, centroids)
        if did_converge(old_centroids, centroids):
            break

    logger.finish()


class Logger(object):
    def __init__(self, output_filename):
        self.f = open(output_filename, 'w', encoding='utf-8')

    def log_iteration_data(self, epoch, centroids):
        output_str = f"[iter {epoch}]:{','.join([str(location) for location in centroids])}"
        print(output_str)
        self.f.write(output_str)
        self.f.write("\n")

    def finish(self):
        self.f.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
