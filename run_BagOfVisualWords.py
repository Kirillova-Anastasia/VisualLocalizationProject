import torch
import torchvision
import cv2
from BagOfVisaulWords import BagOfVisaulWords
from os import listdir
from os.path import join

dataset_dir = '/app/pitts_preparation/datasets/'
test_dir = join(dataset_dir, 'pitts30k', 'images', 'test')

model = BagOfVisaulWords()
model.fit(join(test_dir, 'database'))

with open('predictions.txt', 'w') as f:
    for image_name in tqdm(listdir(join(test_dir, 'queries')), desc='Testing'):
        image = cv2.imread(join(dataset_folder, image_name))
        result = model.predict(image)
        print(result)