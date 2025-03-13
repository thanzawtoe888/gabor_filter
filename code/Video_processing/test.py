import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from glob import glob
from tqdm import tqdm
import subprocess

plt.style.use('ggplot')

input_file = r'D:\Crack-Dataset\my_data\26_2_25_Bending_test_for_Calcined_clay\VID_20250226_133535.mp4'

# Display video using HTML5 video player
display(HTML(f"""
<video width="700" controls>
  <source src="{input_file}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""))