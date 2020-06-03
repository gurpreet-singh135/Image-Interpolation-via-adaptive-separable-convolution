import os
import numpy as np
import matplotlib.pyplot as plt
from cv2.optflow import calcOpticalFlowSF
from PIL import Image

def calc_simple_flow(image1,image2):
    flow = calcOpticalFlowSF(image1, image2, layers=3, averaging_block_size=3, max_flow=4)
    n = np.sum(1 - np.isnan(flow), axis=0)
    n = np.sum(n,axis=0)
    # print(n)
    flow[np.isnan(flow)] = 0
    return np.linalg.norm(np.sum(flow, axis=(0, 1)) / n)

###To Test Simple Flow###

# if __name__=="__main__":
#     image1_path = "/home/z3u5/Downloads/DAVIS-2017-Unsupervised-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/bear/00000.jpg"
#     image2_path = "/home/z3u5/Downloads/DAVIS-2017-Unsupervised-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/bear/00001.jpg"
#     image1 = np.array(Image.open(image1_path))
#     image2 = np.array(Image.open(image2_path))
#     flow = calc_simple_flow(image1,image2)
#     print(flow)