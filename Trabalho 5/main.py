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

INPUT_IMAGE = [r"./img/0.bmp",r"./img/1.bmp",r"./img/2.bmp",r"./img/3.bmp",r"./img/4.bmp",r"./img/5.bmp",r"./img/6.bmp",r"./img/7.bmp",r"./img/8.bmp"]
# INPUT_IMAGE = [r"./img/9.bmp"] 
HUE_MIN = 70
HUE_MAX = 170
HUE_MIN_RAMP = HUE_MIN + (HUE_MIN + HUE_MAX)/8
HUE_MAX_RAMP = HUE_MAX - (HUE_MIN + HUE_MAX)/8
SAT_MIN = 10
SAT_MAX = 50
VAL_MIN = 10
VAL_MAX = 60
T_LOW = 0.3
T_HIGH = 0.95

#===============================================================================
def hue_calc(x):
    """           _________     
                 /         \
        ________/           \____________
    """

    # return np.where((x > HUE_MIN) & (x < HUE_MAX),
    #                 ((((HUE_MIN+HUE_MAX)/2)-2*np.absolute(((HUE_MIN+HUE_MAX)/2)-x)))/((HUE_MIN+HUE_MAX)/2)
    #                 ,0)

    return np.where(x < HUE_MIN,
                    0, 
                    np.where(x < HUE_MIN_RAMP,
                            ((((HUE_MIN+HUE_MAX)/2)-2*np.absolute(((HUE_MIN+HUE_MAX)/2)-x)))/((HUE_MIN+HUE_MAX)/2), 
                            np.where(x<HUE_MAX_RAMP,
                                     1,
                                     np.where(x < HUE_MAX,
                                              ((((HUE_MIN+HUE_MAX)/2)-2*np.absolute(((HUE_MIN+HUE_MAX)/2)-x)))/((HUE_MIN+HUE_MAX)/2),
                                              0))))


def sat_calc(x):
    """
                  _________     
                 /        
        ________/           
    """
    x*=100
    return np.where((x > SAT_MIN) & (x < SAT_MAX),
                    (1/(SAT_MAX-SAT_MIN))*x -(SAT_MIN/(SAT_MAX-SAT_MIN)) ,
                    np.where(x < SAT_MIN, 0, 1))

def val_calc(x):
    """
                  _________     
                 /        
        ________/           
    """
    x*=100
    return np.where((x > VAL_MIN) & (x < VAL_MAX) , (1/(VAL_MAX-VAL_MIN))*x -(VAL_MIN/(VAL_MAX-VAL_MIN)) , np.where(x < VAL_MIN, 0 , 1))

def createMask(img,bg):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Redimensiona o fundo para se adequar a imagem
    bg = cv2.resize(bg, (int(img.shape[1]), int(img.shape[0])))

    alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha = alpha.astype (np.float32)/255
    alpha_n = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha_n = alpha_n.astype (np.float32)/255

    # Criando mascara
    h = hue_calc(imgHSV[:,:,0])
    s = sat_calc(imgHSV[:,:,1])
    v = val_calc(imgHSV[:,:,2])   
    alpha = (h*s*v)

    # Normalizando
    cv2.normalize(alpha, alpha_n, 0, 1, cv2.NORM_MINMAX)    
    
    # Negativa
    alpha_n = (1.0) - alpha

    # Threshold fundo
    alpha_n = np.where(alpha_n < T_LOW, 0 , alpha_n)
    alpha_n = np.where(alpha_n > T_HIGH, 1 , alpha_n)

    alpha_color = np.zeros(img.shape)
    #B
    alpha_color[:,:,0] = np.where(alpha_n < T_LOW, 1 , 0)
    #G
    alpha_color[:,:,1] = np.where(alpha_n > T_HIGH, 1 , 0)  
    #R
    alpha_color[:,:,2] = 1 - (alpha_color[:,:,0] + alpha_color[:,:,1])
 
    cv2.imshow("img_ANTES",img)
    img[:,:,1] = np.where( alpha_color[:,:,2] == 1, (img[:,:,0]+img[:,:,2])/2, img[:,:,1])

    # Chroma key
    res = np.zeros(img.shape)
    res[:,:,0] = alpha_n[:,:]*img[:,:,0] + bg[:,:,0]*(1-alpha_n[:,:])
    res[:,:,1] = alpha_n[:,:]*img[:,:,1] + bg[:,:,1]*(1-alpha_n[:,:])
    res[:,:,2] = alpha_n[:,:]*img[:,:,2] + bg[:,:,2]*(1-alpha_n[:,:])
    
    res[:,:,1] = np.where( alpha_color[:,:,2] == 1, (res[:,:,0]+res[:,:,2])/2, res[:,:,1])
    
    cv2.imshow("alpha_color",alpha_color)    
    #cv2.imshow("alpha_n",alpha_n)
    #cv2.imshow("img_teste",img)
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
