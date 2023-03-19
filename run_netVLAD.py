import math
import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import datasets_ws
import math
import torch
import torchvision
import test
import logging
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler
from IPython.display import clear_output
import warnings
from NetVLAD import GeoLocalizationNet
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

dataset_dir = '/app/pitts_preparation/datasets/'

triplets_ds = datasets_ws.TripletsDataset(dataset_dir, 'pitts30k/', "train", negs_num_per_query = 10)
val_ds = datasets_ws.BaseDataset(dataset_dir, 'pitts30k/', "val")
test_ds = datasets_ws.BaseDataset(dataset_dir, 'pitts30k/', "test")


model = GeoLocalizationNet()
model = model.to(device)

triplets_ds.is_inference = True
model.aggregation.initialize_netvlad_layer(triplets_ds, model.backbone)

optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, weight_decay=0.001)
criterion_triplet = nn.TripletMarginLoss(margin=0.1, p=2, reduction="sum")

epochs = 10


batch_losses = []
triplet_losses = []
rec1 = []
rec5 = []

best_r5 = start_epoch_num = not_improved_num = 0


for epoch in tqdm(range(epochs), desc = "Epochs"):
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    
    
    num_loops = 1
    for loop in range(num_loops):
        print(f'loop : {loop+1}')
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(model)
        triplets_ds.is_inference = False
        
        triplets_dl = DataLoader(dataset=triplets_ds,
                                 batch_size=4,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=("cuda" == "cuda"),
                                 drop_last=True)
        
        model = model.train()
        
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, desc = 'Batch processing'):
            
            # Flip all triplets or none
            # if args.horizontal_flip:
            #     images = transforms.RandomHorizontalFlip()(images)
            
            # Compute features of all images (images contains queries, positives and negatives)
            features = model(images.to("cuda"))
            loss_triplet = 0
            
            # if args.criterion == "triplet":
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(4, 10, 3), 1, 0)
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T
                loss_triplet += criterion_triplet(features[queries_indexes],
                                                  features[positives_indexes],
                                                  features[negatives_indexes])
            # elif args.criterion == 'sare_joint':
            #     # sare_joint needs to receive all the negatives at once
            #     triplet_index_batch = triplets_local_indexes.view(args.train_batch_size, 10, 3)
            #     for batch_triplet_index in triplet_index_batch:
            #         q = features[batch_triplet_index[0, 0]].unsqueeze(0)  # obtain query as tensor of shape 1xn_features
            #         p = features[batch_triplet_index[0, 1]].unsqueeze(0)  # obtain positive as tensor of shape 1xn_features
            #         n = features[batch_triplet_index[:, 2]]               # obtain negatives as tensor of shape 10xn_features
            #         loss_triplet += criterion_triplet(q, p, n)
            # elif args.criterion == "sare_ind":
            #     for triplet in triplets_local_indexes:
            #         # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
            #         q_i, p_i, n_i = triplet
            #         loss_triplet += criterion_triplet(features[q_i:q_i+1], features[p_i:p_i+1], features[n_i:n_i+1])
            
            del features
            loss_triplet /= (4 * 10)
            
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()
            
            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            
            del loss_triplet
        triplet_losses.append(epoch_losses.mean())
        batch_losses.append(batch_loss)
        # print(f"Batch loss: {batch_loss:.4f}")
        # print(f"Average triplet loss: {epoch_losses.mean():.4f}")
        
        
            
    recalls, recalls_str = test.test(val_ds, model)
    rec1.append(recalls[0])
    rec5.append(recalls[1])
    clear_output()
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.set_title('Loss')
    ax0.plot(batch_losses, label = 'Batch')
    ax0.plot(triplet_losses, label = 'Average Triplet loss')
    ax0.set_xlabel('Epoch')
    ax0.legend()
    ax1.set_title('Recall')
    ax1.plot(rec1, label = 'R@1')
    ax1.plot(rec5, label = 'R@5')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    plt.legend()
    # plt.show()
    plt.savefig('current_training_plot.png')
    is_best = recalls[1] > best_r5
    # print(f"Recalls on val set {val_ds}: {recalls_str}")