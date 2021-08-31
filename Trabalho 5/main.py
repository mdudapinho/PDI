#===============================================================================
# Trabalho 5
#-------------------------------------------------------------------------------
# Equipe:
# Lucas Ricardo Marques de Souza
# Maria Eduarda Pinho
#===============================================================================

from math import sqrt
import numpy as np
import statistics
import timeit
import sys
import cv2
from scipy import signal
import matplotlib.pyplot as plt

#===============================================================================

#INPUT_IMAGE = [r"./img/0.bmp",r"./img/1.bmp",r"./img/2.bmp",r"./img/3.bmp",r"./img/4.bmp",r"./img/5.bmp",r"./img/7.bmp",r"./img/8.bmp"]
INPUT_IMAGE = [r"./img/2.bmp"] #,r"./img/1.bmp",r"./img/2.bmp",r"./img/3.bmp",r"./img/4.bmp",r"./img/5.bmp",r"./img/7.bmp",r"./img/8.bmp"]

HUE_MIN = 70
HUE_MAX = 170
SAT = 75
LUM = 20

#===============================================================================
def hue_calc(x):
    return np.where((x > HUE_MIN) & (x < HUE_MAX), ((HUE_MIN+HUE_MAX)/2)-2*np.absolute(((HUE_MIN+HUE_MAX)/2)-x), 0)

def sat_calc(x):
    x*=100
    return np.where(x > SAT, (1/(100-SAT))*x -(SAT/(100-SAT)) , 0)

def lum_calc(x):
    x*=100
    return np.where(x > LUM, (1/(100-LUM))*x -(LUM/(100-LUM)) , 0)

def createMask(img,bg):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Redimensiona o fundo para se adequar a imagem
    bg = cv2.resize(bg, (int(img.shape[1]), int(img.shape[0])))

    res = np.zeros(img.shape)
    alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha_n = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Criando mascara
    h = hue_calc(imgHLS[:,:,0])
    l = lum_calc(imgHLS[:,:,1])
    s = sat_calc(imgHLS[:,:,2])
    alpha = (h*s*l)

    alpha = np.where(alpha > 1, 1, alpha)

    # Normalizando
    # cv2.normalize(alpha, alpha_n, 0, 1, cv2.NORM_MINMAX)
    alpha_n = 1 - alpha

    # Chroma key
    res[:,:,0] = alpha_n[:,:]*img[:,:,0] + bg[:,:,0]*(1-alpha_n[:,:])
    res[:,:,1] = alpha_n[:,:]*img[:,:,1] + bg[:,:,1]*(1-alpha_n[:,:])
    res[:,:,2] = alpha_n[:,:]*img[:,:,2] + bg[:,:,2]*(1-alpha_n[:,:])

    cv2.imshow("alpha_n",alpha_n)
    cv2.imshow("teste",res)
    cv2.waitKey()
    return res

def main():

    for img in INPUT_IMAGE:
        bg = cv2.imread(r"./img/bg.bmp")
        print("imagem:", img)
        imagem = cv2.imread(img)
        if imagem is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # Filtro de verde
        
        bg = bg.astype (np.float32) / 255
        imagem = imagem.astype (np.float32) / 255

        #imagem = cv2.resize(imagem, (int(imagem.shape[1]/2), int(imagem.shape[0]/2)))

        start_time = timeit.default_timer ()
        chroma = createMask(imagem, bg)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))

        cv2.imwrite (img+'_out.png', chroma*255)

    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
