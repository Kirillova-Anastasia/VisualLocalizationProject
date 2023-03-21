import torch
import torchvision
import cv2
from BagOfVisaulWords import BagOfVisaulWords
from os import listdir
from os.path import join
from tqdm import tqdm

keypoints_algorithm = 'orb'

dataset_dir = '/app/pitts_preparation/datasets/'
test_dir = join(dataset_dir, 'pitts30k', 'images', 'test')

model = BagOfVisaulWords(keypoints_algorithm=keypoints_algorithm)
model.fit(join(test_dir, 'database'))

with open('predictions.txt', 'w') as f:
    image_names = listdir(join(test_dir, 'queries'))
    with open('queries_file_names_with_indexes.txt', 'w') as f2:
        for i, file_name in enumerate(image_names):
            f2.write(f'{i} {file_name}\n')
    
    for i, image_name in tqdm(enumerate(image_names), desc='Testing'):
        image = cv2.imread(join(test_dir, 'queries', image_name))
        result = model.predict(image)
        print(i, result, file=f)