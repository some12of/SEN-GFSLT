
from PIL import Image
from PIL import ImageEnhance
import PIL
import random
import numpy as np

class Brightness(object):#用于调整图像的亮度
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip): #定义 __call__ 方法，允许将类实例当作函数调用。该方法接受一个参数 clip，表示一组图像（可以是PIL图像或NumPy数组）。
        factor = random.uniform(self.min, self.max) #从指定的最小和最大亮度因子之间随机选择一个亮度因子。
        if isinstance(clip[0], PIL.Image.Image): #检查输入图像是否为PIL图像
            im_w, im_h = clip[0].size #获取其宽度和高度。
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0]))) #抛出类型错误，提示期望输入NumPy数组或PIL图像，但得到了列表
        new_clip = [] #用于存储调整后的图像。
        for img in clip: #遍历输入的图像
            enh_bri = ImageEnhance.Brightness(img) #创建一个亮度增强对象
            new_img = enh_bri.enhance(factor=factor) #使用增强对象调整图像的亮度。
            new_clip.append(new_img)
        return new_clip

class Color(object): #用于调整图像的饱和度
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_col = ImageEnhance.Color(img)
            new_img = enh_col.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip

class Contrast(object): #用于调整图像的对比度
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_con = ImageEnhance.Contrast(img)
            new_img = enh_con.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip

class Sharpness(object): #用于调整图像的锐度
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_sha = ImageEnhance.Sharpness(img)
            new_img = enh_sha.enhance(factor=1.5)
            new_clip.append(new_img)
        return new_clip