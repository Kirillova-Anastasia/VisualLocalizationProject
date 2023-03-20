import math
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import faiss
import cv2
from os import listdir
from os.path import join
import pandas as pd


class BagOfVisaulWords(nn.Module):
    """BagOfVisaulWords implementation"""

    def __init__(self, clusters_num=64):
        """
        Args:
            clusters_num : int
                The number of clusters
            status : str
                TBP
        """
        super().__init__()
        self.extractor = cv2.SIFT_create()
        self.batch_size = 1000
        self.kmeans = MiniBatchKMeans(
            n_clusters=clusters_num, 
            random_state=42, 
            batch_size=self.batch_size,
            n_init='auto'
        )
        self.neighbor = NearestNeighbors(n_neighbors = 5)


    def _get_features(self, image):
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return keypoints, descriptors


    def _build_histogram(self, descriptor_list):
        histogram = np.zeros(len(self.kmeans.cluster_centers_))
        cluster_result = self.kmeans.predict(descriptor_list.astype('double'))
        for i in cluster_result:
            histogram[i] += 1.0
        return histogram


    def fit(self, dataset_folder):
        image_names = listdir(dataset_folder)
        is_good = []

        descriptor_list = []
        for image_name in tqdm(image_names, desc = 'Getting descriptors'):
            image = cv2.imread(join(dataset_folder, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            keypoint, descriptor = self._get_features(image)
            if descriptor is not None:
                descriptor_list.append(descriptor)
                is_good.append(True)
            else:
                is_good.append(False)
            
            if len(descriptor_list) == self.batch_size:
                self.kmeans = self.kmeans.partial_fit(np.vstack(descriptor_list).astype('double'))
                print('Kmeans partially fitted')
                descriptor_list.clear()
        if len(descriptor_list) > 0:
            self.kmeans = self.kmeans.partial_fit(np.vstack(descriptor_list).astype('double'))
        
        found_cnt = np.sum(np.array(is_good) == True)
        print(f'I found descriptors for {found_cnt}/{len(image_names)} images in database')

        with open('database_file_names_with_indexes.txt', 'w') as f:
            f.write(f'index file_name is_good\n')
            for i, file_name in enumerate(image_names):
                if i >= len(is_good):
                    break
                f.write(f'{i} {file_name} {is_good[i]}\n')

        preprocessed_images = []
        for image_name in tqdm(image_names, desc = 'Getting histograms'):
            image = cv2.imread(join(dataset_folder, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoint, descriptor = self._get_features(image)

            if descriptor is not None:
                histogram = self._build_histogram(descriptor)
                preprocessed_images.append(histogram)
        self.neighbor.fit(preprocessed_images)

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoint, descriptor = self._get_features(image)
        histogram = self._build_histogram(descriptor)
        dist, result = self.neighbor.kneighbors([histogram])
        return result