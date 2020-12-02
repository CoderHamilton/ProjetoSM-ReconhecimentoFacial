# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:54:58 2020

@author: hamil
"""

from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os import isdir
from numpy import asarray

detector = MTCNN()

def extrair_face(arquivo, size = (160, 160)):
    
    img = Image.open(arquivo)
    img = img.convert('RGB')
    array = asarray(img)
    result = detector.detect_faces(array)
    
    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height
    
    face = array[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(size)
    
    return image

def load_fotos(diretorio_src, diretorio_target):
    print(diretorio_src)
    print(diretorio_target)

def load_diretorio(diretorio_src, diretorio_target):
    
    for subdir in listdir(diretorio_src):
        
        path = diretorio_src + subdir + "\\"
        
        path_tg = diretorio_target + subdir + "\\"
        
        if not isdir(path):
            continue
        
        load_fotos(path, path_tg)

if __name__ == '__main__':
    
    load_diretorio(
        "D:\\ProjectMultimidia\\ProjectMultimidia\\data\\fotos\\",
        "D:\\ProjectMultimidia\\ProjectMultimidia\\data\\faces\\"
        )