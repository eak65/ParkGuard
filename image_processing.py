import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def show_image(title, image):
    plt.imshow(image)
    plt.axis('on')  
    plt.title(title)  
    plt.show()
 
def cv_show_image(title, image):
    channels=image.shape[-1]
    if channels==3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    cv2.imshow(title,image)
    cv2.waitKey(0)
 
def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2: 
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    rgb_image = resize_image(rgb_image,resize_height,resize_width)
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image

def resize_image(image,resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]
    if (resize_height is None) and (resize_width is None):
        return image
    if resize_height is None:
        resize_height=int(height*resize_width/width)
    elif resize_width is None:
        resize_width=int(width*resize_height/height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image
def scale_image(image,scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image,dsize=None, fx=scale[0],fy=scale[1])
    return image
 
 
 
def show_image_rect(win_name,image,rect):
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h=rect
    point1=(x,y)
    point2=(x+w,y+h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)
 
def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
 
def save_image(image_path, rgb_image,toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2: 
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)
 
def combime_save_image(orig_image, dest_image, out_dir,name,prefix):
    dest_path = os.path.join(out_dir, name + "_"+prefix+".jpg")
    save_image(dest_path, dest_image)
 
    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name,prefix)), dest_image)