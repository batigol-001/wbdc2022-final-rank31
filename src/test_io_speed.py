import os
import time
import numpy as np
import cv2
import zipfile
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, RandomResizedCrop, \
    RandomHorizontalFlip

if __name__ == "__main__":
    print("test PIL.........")
    start_time = time.time()
    for i in range(1000):
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
    print("cv:sum time:%.5f" % (end_time - start_time))

    print("test opencv.........")
    start_time = time.time()
    for i in range(1000):
        vid = "13438094804486527000"
        zip_path = os.path.join("/home/tione/notebook/data/zip_frames/unlabeled", f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())
        img_content = handler.read(namelist[0])
        cv_img = cv2.imdecode(np.frombuffer(img_content, np.uint8), 1)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img = cv2.GaussianBlur(cv_img, (3, 3), 3)
        cv_img = cv2.resize(cv_img, (256, 256))
        cv_img = cv_img[16:240, 16:240]
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        cv_out = transform(cv_img)

    end_time = time.time()
    print("cv:sum time:%.5f" % (end_time - start_time))


import numpy as np
import pandas as pd

df1 = pd.read_csv("result.csv", index_col=False)
df2 = pd.read_csv("result_1.csv", index_col=False)
df1.columns = ["id", "pred1"]
df2.columns = ["id", "pred2"]
df = df1.merge(df2)
print(df.loc[df.pred1!=df.pred2].shape[0] / len(df))






