#===============================================================================
# Trabalho 3 
#-------------------------------------------------------------------------------
# Equipe:
# Lucas Ricardo Marques de Souza
# Maria Eduarda Pinho
#===============================================================================

import sys
import timeit
import numpy as np
import cv2
from math import sqrt

#===============================================================================

INPUT_IMAGE =  r"./Wind Waker GC.bmp"
# INPUT_IMAGE =  r"GT2.BMP"

SIGMA = 10
REP_GAUSSIAN = 4
REP_BOXBLUR = 3
THRESHOLD = 128

ALFA = 0.8
BETA = 0.2
#===============================================================================

def gaussian(img, mask):
    """
    Implementacao bloom com o filtro da Gaussiana.
    
    Parametros:
        img: Imagem original BGR.
        mask: Mascara a ser utilizada apos passar pelo brightPass.

    Variaveis:
        mask_blur: Inicialmente recebe zeros no formato da imagem.
                Esta e a variavel que recebe a mascara borrada
        SIGMA: Fator interno na funcao cv2.GaussianBlur.
        REP_GAUSSIAN: Macro com a quantidade de repeticoes do filtro gaussiana.
    """
    # Inicializa blur com zeros
    mask_blur = np.zeros(img.shape, np.float32)

    # Itera borrando a mascara com a gaussiana
    for exp in range(REP_GAUSSIAN):
        mask_blur = cv2.add(mask_blur,cv2.GaussianBlur(mask, (0,0), SIGMA*pow(2,exp)))

    # Soma na imagem final
    img_blur = cv2.addWeighted(img,ALFA,mask_blur,BETA,0)

    return img_blur

def boxBlur(img, mask):
    """
    Implementacao bloom com o filtro da media.
    
    Parametros:
        img: Imagem original BGR
        mask: Mascara a ser utilizada apos passar pelo brightPass.

    Variaveis:
        mask_blur: Inicialmente recebe zeros no formato da imagem.
                Esta e a variavel que recebe a mascara borrada
        SIGMA: Fator interno na funcao cv2.GaussianBlur.
        REP_GAUSSIAN: Macro com a quantidade de repeticoes do filtro gaussiana.
    """ 
    # Inicializa blur com zeros
    mask_blur = np.zeros(img.shape, np.float32)

    for exp in range(REP_GAUSSIAN):
        mask_blur = cv2.add(mask_blur,gaussianBoxBlur(mask, SIGMA*pow(2,exp)))

    # Somar na imagem 
    img_blur = cv2.addWeighted(img,ALFA,mask_blur,BETA,0)

    return img_blur

def janelaIdeal(sigma):
    """
    Calculo da janela ideal segundo o artigo:
    https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf

    Parametros:
        sigma: Sigma usado na GaussianBlur
    """
    return int(sqrt(12*(sigma**2)/REP_BOXBLUR+1))

def gaussianBoxBlur(mask, sigma):
    """
    Simulacao do GaussianBlur com filtro da media. 
    
    Parametros:
        sigma: Sigma usado para calcular a janela ideal
        mask: Mascara a ser utilizada apos passar pelo brightPass.
    """ 
    janela = janelaIdeal(sigma)

    for _ in range(REP_BOXBLUR):
        mask = cv2.blur(mask,(janela, janela))
    
    return mask

def createMask(img):
    """
    Funcao que cria a mascara com base na luminancia. 
    
    Parametros:
        img: Imagem de entrada

    Variaveis:
        imgHLS: Recebe a imagem HLS.
        Lchannel: Recebe os valores de luminancia.
        mask: Recebe a imagem somente com as fontes de luz.
        res: Recebe o equivalente dos pixels na imagem original, 
            mas com base na mascara, agora em BGR
    """
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    Lchannel = imgHLS[:,:,1]
    mask = cv2.inRange(Lchannel, THRESHOLD/255, 1)
    res = cv2.bitwise_and(img,img, mask= mask)

    return res

def main():
    # Abre a imagem.
    imgBGR = cv2.imread(INPUT_IMAGE)
    if imgBGR is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # Normalizando com float
    imgBGR = imgBGR.astype (np.float32) / 255

    # Filtro da luminancia
    brightPass = createMask(imgBGR)
    
    # Inicio da funcao gaussiana
    start_time = timeit.default_timer ()
    print("Gaussiano:")
    img_gaussian = gaussian(imgBGR, brightPass)
    print ('\tTempo: %f' % (timeit.default_timer () - start_time))

    # Inicio da funcao filtro da media
    start_time = timeit.default_timer ()
    print("Filtro da média:")
    img_blur = boxBlur(imgBGR, brightPass)
    print ('\tTempo: %f' % (timeit.default_timer () - start_time)) 
    
    cv2.imshow ('Filtro da luminancia', brightPass)
    cv2.imshow ('Gaussiana', img_gaussian)
    cv2.imshow ('Filtro da media', img_blur)
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()