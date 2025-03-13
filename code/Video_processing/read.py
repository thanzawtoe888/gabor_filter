import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from glob import glob

import IPython.display as ipd
from tqdm import tqdm

import subprocess

plt.style.use('ggplot')

input_file = r'D:\Crack-Dataset\my_data\26_2_25_Bending_test_for_Calcined_clay\VID_20250226_133535.mp4'

ipd.Video('input_file', width=700, embed=True)
plt.show()