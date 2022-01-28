from PIL import Image
import pyheif
import glob
from PIL import Image
import os

def conv(image_path):
    new_name = image_path.replace('JPG', 'jpeg')
    heif_file = pyheif.read(image_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
        )
    data.save(new_name, "jpeg")

def JPGtojpeg(data):
    im = Image.open(data)
    im.save(str(idx) +".jpeg", "jpeg")

    
def resolution(image_path, idx):
    size = 512, 512
    im = Image.open(image_path)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(str(idx) +".jpeg", "jpeg")

lst = glob.glob("*.jpeg")
# lst = glob.glob("*.JPG")
for idx, l in enumerate(lst):
    # conv(l)
    resolution(l, idx)
    # JPGtojpeg(l)
