import math
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import faiss
import cv2
from os import listdir
from os.path import join


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


class BagOfVisaulWords(nn.Module):
    """BagOfVisaulWords implementation"""

    def __init__(self, clusters_num=64):
        """
        Args:
            clusters_num : int
                The number of clusters
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.extractor = cv2.SIFT_create()
        self.kmeans = KMeans(n_clusters = clusters_num)
        self.neighbor = NearestNeighbors(n_neighbors = 20)

    def _get_features(self, image):
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return keypoints, descriptors

    def _build_histogram(self, descriptor_list):
        cluster_alg = self.kmeans
        histogram = np.zeros(len(cluster_alg.cluster_centers_))
        cluster_result = cluster_alg.predict(descriptor_list)
        for i in cluster_result:
            histogram[i] += 1.0
        return histogram

    def fit(self, dataset_folder):
        descriptor_list = []
        i = 0
        for image_name in tqdm(listdir(dataset_folder), desc = 'Getting descriptors'):
            image = cv2.imread(join(dataset_folder, image_name))
            keypoint, descriptor = self._get_features(image)
            descriptor_list.append(descriptor)
            i += 1
            if i == 10:
                break
        for descriptor in descriptor_list:
            print(descriptor.shape)
        self.kmeans.fit(descriptor_list)

        preprocessed_images = []
        for descriptor in tqdm(descriptor_list, desc = 'Getting histograms'):
            if (descriptor is not None):
                histogram = self._build_histogram(descriptor)
                preprocessed_images.append(histogram)
            i += 1
            if i == 20:
                break
        self.neighbor.fit(preprocessed_images)

    def predict(self, image):
        keypoint, descriptor = self._get_features(image)
        histogram = self._build_histogram(descriptor)
        dist, result = neighbor.kneighbors([histogram])
        return result