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

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from utils.utils import dice, resample_3d

import open_clip

from monai.inferers import sliding_window_inference
from models.swin_gcis import SwinAP, SwinSingle

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--model", default="./runs/test/model.pt", type=str, help="trained checkpoint path"
)
parser.add_argument("--display_path", action="store_true", help="show the path index in different automatic pathway modules")
parser.add_argument("--use_prompt", action="store_true", help="text prompt input and use automatic pathway")
parser.add_argument("--ap_num", default=6, type=int, help="number of pathway")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_gradcheckpoint", action="store_true", help="use gradient checkpointing to save memory")

def is_ras(header):
    ornt = axcodes2ornt(header.get_axis_codes())
    return not np.all(np.equal(ornt, np.arange(6)))

def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_prompt:
        model = SwinAP(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            ap_num=args.ap_num,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
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
            dropout_path_rate=0.0,
            use_checkpoint=args.use_gradcheckpoint,
        )
    model_dict = torch.load(args.model)["state_dict"]
    if "module." in list(model_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(model_dict.keys()):
            model_dict[key.replace("module.", "")] = model_dict.pop(key)
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    prompt_model, tokenizer = None, None
    if args.use_prompt:
        ### load prompt network
        prompt_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device, cache_dir="/path/to/clip_weights")
        tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            if args.use_prompt:
                val_inputs, val_labels, text_prompt = (batch["image"].cuda(), batch["label"].cuda(), batch["prompt"])
                text_tokens = tokenizer(text_prompt).cuda()
                text_features = prompt_model.encode_text(text_tokens)
                text_features = text_features.float()
            else:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            if args.use_prompt:
                val_outputs = sliding_window_inference(
                    val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian", prompt_in=text_features, display_path=args.display_path
                )
            else:
                val_outputs = sliding_window_inference(
                    val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
                )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            dice_list_sub = []
            for i in range(1, args.out_channels):
                Dice_curclass = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(Dice_curclass)
            mean_dice = np.mean(dice_list_sub)
            if args.out_channels > 2:
                print("Mean Dice: {} for {} classes".format(mean_dice, args.out_channels-1))
            else:
                print("Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
            )

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == "__main__":
    main()
