# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:test.py
    @ide:PyCharm
    @time:2019-01-07 20:31
    @author:Sun
"""
from Init import *
from func import DrawPicture

no=10
img=test_images[no]/255.0
DrawPicture(img)