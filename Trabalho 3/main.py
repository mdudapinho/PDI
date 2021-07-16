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

# INPUT_IMAGE =  r"./Wind Waker GC.bmp"
INPUT_IMAGE =  r"GT2.BMP"
SIGMA = 7
JANELA = 15
REP = 3
REP_BOXBLUR = 3
THRESHOLD = 128
ALFA = 0.8
BETA = 0.2

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
            if(dif <= LIMITE):
                img_compare[l][c] = 1
                sum += 1
    sum = sum/(linha*coluna)
    print("\t\tAs imagens sao ", round(sum*100, 2), "% parecidas")

    return img_compare

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
    cv2.imshow ('img_gaussian',img_gaussian)
    cv2.imshow ('img_blur',img_blur)
    cv2.imshow('comparador',comparador(img_gaussian,img_blur))
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