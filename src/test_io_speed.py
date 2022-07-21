import os
import time
import numpy as np
import cv2
import zipfile
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, RandomResizedCrop, \
    RandomHorizontalFlip, ConvertImageDtype
from torchvision.io import read_image


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


if __name__ == "__main__":
    print("test PIL.........")
    start_time = time.time()
    for i in range(1):
        vid = "13438094804486527000"
        zip_path = os.path.join("/home/tione/notebook/data/zip_frames/unlabeled/", f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())
        img_content = handler.read(namelist[0])

        pil_img = Image.open(BytesIO(img_content))
        transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pil_out = transform(pil_img)

    end_time = time.time()
    print("test PIL:sum time:%.5f" % (end_time - start_time))

    print("test opencv.........")
    start_time = time.time()
    total_s1 = 0
    total_s2 = 0.0
    total_s3 = 0.0
    for i in range(1):
        vid = "13438094804486527000"
        zip_path = os.path.join("/home/tione/notebook/data/zip_frames/unlabeled", f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())
        img_content = handler.read(namelist[0])

        cv_img = cv2.imdecode(np.frombuffer(img_content, np.uint8), 1)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img = cv2.resize(cv_img, (256, 256))
        cv_img = cv_img[16:240, 16:240, :]

        cv_img_tensor = torch.from_numpy(cv_img)
        cv_img_tensor = cv_img_tensor.permute(2, 0, 1)
        transform = Compose([
            ZeroOneNormalize(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        out = transform(cv_img_tensor)

    end_time = time.time()
    print("test opencv:sum time:%.5f" % (end_time - start_time))
    print(total_s1, total_s2, total_s3)
    pil_out = pil_out.view(-1)
    cv_out = out.contiguous().view(-1)
    print(pil_out)
    print(cv_out)
    print(pil_out.shape, cv_out.shape)
    print(sum(pil_out == cv_out) / len(pil_out))







