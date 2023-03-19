import math
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import faiss

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        # print(f': x.shape : {x.size()}')
        # print(f'conv(x) size = {self.conv(x).size()}')
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, cluster_ds, backbone):
        descriptors_num = 100000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, 
                                batch_size=16, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            # logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to("cuda")
                outputs = backbone(inputs)
                # print(f'outputs: {outputs.size()}')
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                # print(f'norm_outputs: {norm_outputs.size()}')
                # print(f'shape [0] : {norm_outputs.shape[0]}')
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * 16 * descs_num_per_image
                # print(f'image_descriptors : {image_descriptors.shape}')
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
#         logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to("cuda")


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]


def get_backbone(model_name):
    # The aggregation layer works differently based on the type of architecture
    if model_name.startswith("resnet"):
        if 1 == 2:
            backbone = get_pretrained_model(args)
        elif model_name.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif model_name.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif model_name.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if model_name.endswith("conv4"):
            # logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif model_name.endswith("conv5"):
            # logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif model_name == "vgg16":
        if 1 == 2:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        # logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif model_name == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        # logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif model_name.startswith("cct"):
        if model_name.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if 1 == 2:
            # logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if 1 == 2:
            # logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        features_dim = 384
        return backbone
    elif model_name.startswith("vit"):
        # assert args.resize[0] in [224, 384], f'Image size for ViT must be either 224 or 384, but it\'s {args.resize[0]}'
        if 1 == 2:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif 1 == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')

        if 1 == 2:
            # logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if 1 == 2:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if 1 == 2:
                    for params in child.parameters():
                        params.requires_grad = True
        backbone = VitWrapper(backbone, args.aggregation)
        
        features_dim = 768
        return backbone

    backbone = torch.nn.Sequential(*layers)
    features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    # print(features_dim)
    return backbone

features_dim = 256

class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self):
        super().__init__()
        self.backbone = get_backbone('resnet18conv4')
        # self.arch_name = 
        self.aggregation = NetVLAD(clusters_num=64, dim=features_dim,
                                   work_with_tokens=False)
    def forward(self, x):
        # print(f'size in forward before backbone: {x.size()}')
        x = self.backbone(x)
        # print(f'size in forward after backbone: {x.size()}')
        x = self.aggregation(x)
        return x