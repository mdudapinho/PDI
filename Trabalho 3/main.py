#===============================================================================
# Trabalho 2 
#-------------------------------------------------------------------------------
# Equipe:
# Lucas Ricardo Marques de Souza
# Maria Eduarda Pinho
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  r"GT2.BMP"  
# Insira apenas valores impares
JANELA = 20
REP = 5
REP_APX = 4

LIMITE = 1.000/255 # Indice que limita a diferenca por pixel no comparador 
 
#===============================================================================

def Gaussian(img, mask):
    
    img_out = cv2.GaussianBlur(mask, (0,0), JANELA)

    for i in rang(REP-1):
        img_out += cv2.GaussianBlur(img_out, (0,0), JANELA * (i+1))

    return img_out

def Blur(img, mask):
    # blur (img, largura, altura,borderType=cv2.BORDER_DEFAULT)
    # borderType seleciona o tipo de tratamento que o opencv vai fazer
    # https://docs.opencv.org/4.5.2/d2/de8/group__core__array.html#gga209f2f4869e304c82d07739337eae7c5afe14c13a4ea8b8e3b3ef399013dbae01
    
    img_out = cv2.blur(mask,(JANELA,JANELA))
    
    for i in rang(REP-1):
        for i in rang(REP_APX):
            img_out += cv2.blur(img_out, JANELA * (i+1), JANELA * (i+1))

    return img_out


def createMask(img):
    return img


def main():
    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()


    mask = createMask()

    img_gaussian = Gaussian(img, mask)
    img_blur = Blur(img, mask)


    cv2.imshow ('img', img)
    cv2.imshow ('mask',mask)
    cv2.imshow ('img_gaussian',img_gaussian)
    cv2.imshow ('img_blur',img_blur)
    cv2.waitKey ()
    cv2.destroyAllWindows ()




def main_Bogdan ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = cv2.resize(img, (int(img.shape[1]*0.8), int(img.shape[0]*0.8)))
    
    v1 = cv2.GaussianBlur(img, (0,0), 20)

    reduzida = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
    
    v2 = cv2.GaussianBlur(reduzida, (0,0), 10)
    reampliada = cv2.resize(v2, (img.shape[1], img.shape[0]))

    cv2.imshow ('img', img)
    cv2.imshow ('v1',v1)
    cv2.imshow ('reduzida',reduzida)
    cv2.imshow ('v2',v2)
    cv2.imshow ('reampliada',reampliada)
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()