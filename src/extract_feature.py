import os
import io
import json
import torch
import zipfile
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from third_party.swin import swin

import random

class RawFrameDataset(Dataset):

    def __init__(self,
                 ann_path: str,
                 zip_frame_dir: str,
                 max_video_frames: int = 32):
        """ This class is used to load raw video frames.
        Args:
            ann_paths (str): the annotation file path.
            zip_frame_dir (str): the directory that saves zip frames.
            max_video_frames (str): the maximum number of video frames.
        """
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.zip_frame_dir = zip_frame_dir
        self.max_video_frames = max_video_frames

        # we follow the common practice as in the ImageNet's preprocessing.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> dict:
        return len(self.anns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Extract the frame tensor from zipped file.
        The output tensor is in shape of [MAX_FRAMES, 3, 224, 224]
        """
        feedid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, feedid[-3:], f'{feedid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        img_name_list = handler.namelist()
        img_name_list = sorted(img_name_list)
        img_name_list = img_name_list[:self.max_video_frames]
        img_tensor = torch.zeros(self.max_video_frames, 3, 224, 224, dtype=torch.half)
        num_frames = len(img_name_list)

        if num_frames <= self.max_video_frames:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:self.max_video_frames]
            select_inds = sorted(select_inds)

        for i, j in enumerate(select_inds):
            i_img_content = handler.read(img_name_list[j])
            i_img = Image.open(io.BytesIO(i_img_content))
            i_img_tensor = self.transform(i_img)
            img_tensor[i, ...] = i_img_tensor
        num_frames = torch.IntTensor([len(img_name_list)])
        return dict(img=img_tensor, num_frames=num_frames)


def parse_args():
    parser = argparse.ArgumentParser("Visual feature extraction")
    parser.add_argument('--zip_frame_dir', type=str, default='/home/tione/notebook/data/zip_frames/unlabeled/')
    parser.add_argument('--ann_path', type=str, default='/home/tione/notebook/data/annotations/unlabeled_new.json')
    parser.add_argument('--swin_pretrained', type=str,
                        default='../opensource_models/swin_base_patch4_window7_224_22k.pth')
    parser.add_argument('--output_path', type=str, default='/home/tione/notebook/data/zip_feats')
    parser.add_argument('--output_file', type=str, default='/home/tione/notebook/data/zip_feats/unlabeled.zip')
    args = parser.parse_args()
    return args


def build_model(swin_pretrained) -> torch.nn.Module:
    """ Load the pretrianed feature extractor (Swin-T here). """
    if not os.path.isfile(swin_pretrained):
        raise IOError(f"Cannot load pretrained swin model from {swin_pretrained}."
                      "Please manually download it from https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth")
    model = swin(swin_pretrained)
    if torch.cuda.is_available():
        model = DataParallel(model.half().cuda(), device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    return model


def main():
    args = parse_args()
    model = build_model(args.swin_pretrained)

    dataset = RawFrameDataset(args.ann_path, args.zip_frame_dir, max_video_frames=8)
    # batch-size == 8 is fine for V100 GPU, please consider use smaller batch-size if OOM issue occurs.
    dataloader = DataLoader(dataset, batch_size=384, num_workers=4, prefetch_factor=8, shuffle=False, pin_memory=True, drop_last=False)
    os.makedirs(args.output_path, exist_ok=True)

    assert not os.path.isfile(args.output_file), f"{args.output_file} already exists. " \
        "If you want to override it, please manually delete this file."

    output_handler = zipfile.ZipFile(args.output_file, 'w', compression=zipfile.ZIP_STORED)

    with torch.no_grad():
        cur = 0
        for dataitem in tqdm(dataloader):
            img, num_frames = dataitem['img'], dataitem['num_frames']
            B, L = img.shape[0:2]
            img = img.view((B * L,) + img.shape[2:])
            # img = img.half()
            feature = model(img)
            feature = feature.view(B, L, -1)
            feature = feature.cpu().numpy().astype(np.float16)
            for i in range(B):
                feedid = dataset.anns[cur]['id']
                ioproxy = io.BytesIO()
                np.save(ioproxy, feature[i, :int(num_frames[i])])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(f'{feedid}.npy', npy_str)
                cur += 1
                if cur % 1000 == 0:
                    print(f"Extract feature {cur}/{len(dataset)}")
    output_handler.close()


if __name__ == '__main__':
    main()
