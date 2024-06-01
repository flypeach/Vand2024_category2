"""Example model file for track 2."""
import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import torch.nn as nn

import open_clip
# from few_shot import memory
from model import LinearLayer
# from dataset import VisaDataset, MVTecDataset, MVTec_locoDataset
from prompt_ensemble_anovl import encode_text_with_prompt_ensemble

from torchvision.transforms import v2

FEATURES_LIST = [6, 12, 18, 24]
IMAGE_SIZE = 518
BACKBONE = 'ViT-L-14-336'
PRETRAINED = 'openai'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OBJ_LIST = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
# OBJ_LIST = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
CONFIG_PATH = './open_clip/model_configs/ViT-L-14-336.json'
FEW_SHOT_FEATURES = [6, 12, 18, 24]
DATASET_NAME = 'mvtec_loco'

# NORMALIZE = {'breakfast_box':[2.9009964342178582,6.668818505980527], 'juice_bottle':[2.7457472026290213,6.352861368225897], 'pushpins':[3.3803797324553266,6.603872872216673], 'screw_bag':[3.958667629421344,6.393484196049201], 'splicing_connectors':[2.5850176484075735,6.489935010015184]}
NORMALIZE = {'breakfast_box':[3.252429754389905,5.452478297993938], 'juice_bottle':[3.033021020337423,5.0260275660305265], 'pushpins':[3.2131757926823106,5.1542454996388685], 'screw_bag':[3.189483816192574,4.8456495609918875], 'splicing_connectors':[3.116475452701868,4.961866544348441]}

class MyModel(nn.Module):
    """Example model class for track 2.

    This class applies few-shot anomaly detection using the WinClip model from Anomalib.
    """

    def __init__(self) -> None:
        super().__init__()
        self.class_name = []
        self.k_shot = 4
        self.mem_features = None
        self.mode = 'test'
        
        # self.transform = transforms.Compose(
        #     [   
        #         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #         transforms.CenterCrop(IMAGE_SIZE),
        #         transforms.ToTensor()
        #     ]
        # )
        
        self.transform = v2.Compose(
            [
                v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            ],
        )
        
        self.model, _, preprocess = open_clip.create_model_and_transforms(BACKBONE, IMAGE_SIZE, pretrained=PRETRAINED)
        self.model.to(DEVICE)
        self.tokenizer = open_clip.get_tokenizer(BACKBONE)
        self.model.eval() 
        
    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        # batch = self.transform(batch)
        # pred_score, anomaly_maps = self.model(batch)
        with open(CONFIG_PATH, 'r') as f:
            model_configs = json.load(f)
            
        trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(FEATURES_LIST), BACKBONE).to(DEVICE)
        
        with torch.cuda.amp.autocast(), torch.no_grad():
            text_prompts, text_prompts_list = encode_text_with_prompt_ensemble(self.model, OBJ_LIST, self.tokenizer, DEVICE)
            
        pred_score = 0
        if self.mode == 'train':
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    image_features, patch_tokens = self.model.encode_image(batch, FEATURES_LIST)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features = []
                    for cls in self.class_name:
                        text_features.append(text_prompts[cls])
                    text_features = torch.stack(text_features, dim=0)

                    # # sample
                    # text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
                    # pr_sp = text_probs[0][1].cpu().item()

                    # pixel
                patch_tokens = trainable_layer(patch_tokens)
                anomaly_maps = []
                for layer in range(len(patch_tokens)):
                    patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=IMAGE_SIZE, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map)
                    
        elif self.mode == 'test':
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features, patch_tokens = self.model.encode_image(batch, FEATURES_LIST)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features = []
                for cls in self.class_name:
                    text_features.append(text_prompts[cls])
                text_features = torch.stack(text_features, dim=0)

                # sample
                text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
                pr_sp = text_probs[0][1].cpu().item()

                # pixel
                patch_tokens = trainable_layer(patch_tokens)
                anomaly_maps = []
                for layer in range(len(patch_tokens)):
                    patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=IMAGE_SIZE, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                anomaly_map = np.sum(anomaly_maps, axis=0)
            
        
                # few shot
                image_features, patch_tokens = self.model.encode_image(batch, FEW_SHOT_FEATURES)
                anomaly_maps_few_shot = []
                for idx, p in enumerate(patch_tokens):
                    if 'ViT' in BACKBONE:
                       p = p[0, 1:, :]
                    else:
                        p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                    cos = pairwise.cosine_similarity(self.mem_features[self.class_name[0]][idx].cpu(), p.cpu())
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = np.min((1 - cos), 0).reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                         size=IMAGE_SIZE, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                anomaly_map = anomaly_map + anomaly_map_few_shot
            
                # normalize
                pr_sp_tmp = np.array(anomaly_map)
                pr_sp_tmp = (pr_sp_tmp - NORMALIZE[self.class_name[0]][0]) / (NORMALIZE[self.class_name[0]][1] - NORMALIZE[self.class_name[0]][0])
            
                pred_score = 0.5 * (pr_sp + pr_sp_tmp)
        
            
        return {"pred_score": pred_score, "anomaly_map": anomaly_map, "anomaly_maps": anomaly_maps, "pr_sp": pr_sp}

    def setup(self, data: dict) -> None:
        """Setup the few-shot samples for the model.

          The evaluation script will call this method to pass the k images for few shot learning and the object class
          name. In the case of MVTec LOCO this will be the dataset category name (e.g. breakfast_box). Please contact
          the organizing committee if if your model requires any additional dataset-related information at setup-time.
          """
        few_shot_samples = data.get("few_shot_samples")
        class_name = data.get("dataset_category")

        self.class_name = class_name
        few_shot_samples = self.transform(few_shot_samples)
        # self.few_shot_samples = few_shot_samples
        self.k_shot = few_shot_samples.shape[0]
        
        features = []
        for i in few_shot_samples.shape[0]:
            image = few_shot_samples[i]
            with torch.no_grad():
                image_features, patch_tokens = self.model.encode_image(image, FEW_SHOT_FEATURES)
                if 'ViT' in BACKBONE:
                    patch_tokens = [p[0, 1:, :] for p in patch_tokens]
                else:
                    patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens)
                
        self.mem_features = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
        
        
        
    def change_classname(self, classname):
        self.class_name = classname
        
        
    def change_mode(self, mode):
        self.mode = mode
        
    
    def change_mem_features(self, mem_features):
        self.mem_features = mem_features
    
    
    
        