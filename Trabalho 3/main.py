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
from math import floor, sqrt

#===============================================================================

# INPUT_IMAGE =  r"./Wind Waker GC.bmp"
INPUT_IMAGE =  r"GT2.BMP"
SIGMA = 10
REP = 4
REP_BOXBLUR = 3
THRESHOLD = 200
ALFA = 0.85
BETA = 0.15

LIMITE = 1.000/255 # Indice que limita a diferenca por pixel no comparador 
#===============================================================================

def comparador(img_1,img_2,):
    sum = 0
    linha, coluna, channels = img_1.shape
    img_compare = np.zeros(img_1.shape)
        
    # Compara pixel a pixel
    for l in range(linha):        
        for c in range(coluna):
            dif = abs(img_1[l][c][0] - img_2[l][c][0]) 
            img_compare[l][c] = dif
            if(dif <= LIMITE):
                #img_compare[l][c] = 1
                sum += 1
    sum = sum/(linha*coluna)
    print("\t\tAs imagens sao ", round(sum*100, 2), "% parecidas")

    return img_compare

def Gaussian(img, mask):
    
    # Inicializa blur com zeros
    mask_blur = np.zeros(img.shape, np.float32)

    for exp in range(REP):
        mask_blur = cv2.add(mask_blur,cv2.GaussianBlur(mask, (0,0), SIGMA*pow(2,exp)))

    # Somar na imagem 
    img_blur = cv2.addWeighted(img,ALFA,mask_blur,BETA,0)

    return img_blur

def Blur(img, mask):

    img_aux = mask

    # Primeira borra
    for i in range(REP_BOXBLUR):
            img_aux = cv2.addWeighted(img_aux,
                                      (1-(1/(i+1))),
                                      cv2.blur(img_aux,(JANELA * (i+1), JANELA * (i+1))),
                                      (1/(i+1)),
                                      0)
    
    img_out = img_aux

    # Borra mais REP-1 vezes
    for _ in range(1,REP):
        for i in range(REP_BOXBLUR):
            img_aux = cv2.addWeighted(img_aux,
                                      (1-(1/(i+1))),
                                      cv2.blur(img_aux,(JANELA * (i+1), JANELA * (i+1))),
                                      (1/(i+1)),
                                      0)
        
        img_out = cv2.add(img_out,img_aux)

    # Somar na imagem 
    img_out = cv2.addWeighted(img,ALFA,img_out,BETA,0)

    return img_out


def createMask(img):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    Lchannel = imgHLS[:,:,1]
    mask = cv2.inRange(Lchannel, THRESHOLD/255, 1)
    res = cv2.bitwise_and(img,img, mask= mask)

    return res


def main():
    # Abre a imagem em escala de cinza.
    imgBGR = cv2.imread(INPUT_IMAGE)
    if imgBGR is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # Normalizando com float
    imgBGR = imgBGR.astype (np.float32) / 255

    brightPass = createMask(imgBGR)
    
    img_gaussian = Gaussian(imgBGR, brightPass)
    img_blur = Blur(imgBGR, brightPass)

    #cv2.imshow ('img_orig', imgBGR)
    #cv2.imshow ('brightPass', brightPass)
    img_gaussian = cv2.resize(img_gaussian, (int(img_gaussian.shape[1]*0.5), int(img_gaussian.shape[0]*0.5)))
    img_blur = cv2.resize(img_blur, (int(img_blur.shape[1]*0.5), int(img_blur.shape[0]*0.5)))
    cv2.imshow ('img_gaussian', img_gaussian)
    cv2.imshow ('img_blur', img_blur)
    #cv2.imshow('comparador',comparador(img_gaussian,img_blur))
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
    try:
        main()
    except KeyboardInterrupt:
        exit()