# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

import open_clip

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from models.swin_gcis import SwinAP, SwinSingle
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="gcis segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--pretrained", default=None, help="use pretrained weights")
parser.add_argument("--ap_model", default=None, help="use automatic pathway weights")
parser.add_argument("--path_selection", default=None, type=str, help="the pathway index in each automatic pathway module, index start from 1")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--use_prompt", action="store_true", help="text prompt input and use automatic pathway")
parser.add_argument("--val_every", default=500, type=int, help="validation frequency")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--ap_num", default=6, type=int, help="number of pathway")
parser.add_argument("--tau", default=1.0, type=float, help="temperature in gumble softmax")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_gradcheckpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--finetune", action="store_true", help="fix the parameter in Swin")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.use_prompt:
        print("gcis training using automatic pathways")
    else:
        print("gcis training using single pathway")
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    # torch.cuda.set_device(args.gpu)
    # torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print("Batch size is:", args.batch_size, ", epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    if args.use_prompt:
        print("pathway number:", args.ap_num, ", tau of gumble softmax:", args.tau)
    if args.use_prompt:
        model = SwinAP(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            ap_num=args.ap_num,
            tau=args.tau,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_gradcheckpoint,
        )
    else:
        model = SwinSingle(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_gradcheckpoint,
        )

    if args.pretrained != None:
        try:
            model_dict = torch.load(args.pretrained)
            state_dict = model_dict["state_dict"]
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            net_state_dict = model.state_dict()
            for key in list(state_dict.keys()):
                new_key = key + "_pre"
                if key in list(net_state_dict.keys()) and state_dict[key].size() != net_state_dict[key].size():
                    print(key + " size not match state dict - fixing!")
                    state_dict[new_key] = state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained weights!")
        except ValueError:
            raise ValueError("Pre-trained weights not available for" + str(args.model_name))
    
    if args.ap_model != None:
        assert not args.use_prompt, "pathway selection training only support the single pathway model"
        assert args.path_selection!=None, "please input the pathway selection"
        args.path_selection = [int(x) for x in args.path_selection.split(",")]
        print("path_selection:", args.path_selection)
        assert len(args.path_selection)==7, "please input right pathway selection. Should be a list with 7 numbers, and each number should be from 1 to ap_num."
        layer_name = [("decoder10.layer_p", "decoder10.layer"), ("decoder5.conv_block_p", "decoder5.conv_block"), ("decoder4.conv_block_p", "decoder4.conv_block"), ("decoder3.conv_block_p", "decoder3.conv_block"), ("decoder2.conv_block_p", "decoder2.conv_block"), ("decoder1.conv_block_p", "decoder1.conv_block"), ("out_p", "out")]
        path_in_position = {}
        for i in range(len(args.path_selection)):
            n = args.path_selection[i]
            assert n >= 1 and n <= args.ap_num, "please input right pathway selection. Should be a list with 7 numbers, and each number should be from 1 to ap_num."
            path_in_position[layer_name[i][0]+str(n-1)] = layer_name[i][1]
        try:
            model_dict = torch.load(args.ap_model)
            state_dict = model_dict["state_dict"]
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            for key in list(state_dict.keys()):
                key_first = key.split(".")[0]
                key_firsttwo = ".".join(key.split(".")[:2])
                for path_name in path_in_position:
                    if path_name == key_first:
                        new_key = path_in_position[path_name] + key[len(key_first):]
                        state_dict[new_key] = state_dict.pop(key)
                    if path_name == key_firsttwo:
                        new_key = path_in_position[path_name] + key[len(key_firsttwo):]
                        state_dict[new_key] = state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)
            print("Using ap model weights!")
        except ValueError:
            raise ValueError("AP model not available for" + str(args.model_name))

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
    
    if args.finetune:
        for n, p in model.named_parameters():
            if "swinViT" in n:
                p.requires_grad = False
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model.cuda()
    model = torch.nn.DataParallel(model)

    prompt_model, tokenizer = None, None
    if args.use_prompt:
        ### load prompt network
        prompt_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device=torch.device('cuda:0'), cache_dir="/path/to/clip_weights")
        tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        prompt_model=prompt_model,
        tokenizer=tokenizer,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
