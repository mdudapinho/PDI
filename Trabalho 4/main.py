#===============================================================================
# Trabalho 4
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

INPUT_IMAGE =  [r"./60.bmp", r"./82.bmp", r"./114.bmp", r"./150.bmp", r"./205.bmp"]
#INPUT_IMAGE =  [r"./60.bmp"]

JANELA_EROSAO = 5
JANELA_DILATACAO = 5

THRESHOLD = 0.2
#===============================================================================

def Erosao(img):
    print("\tErode")

    rows, cols = img.shape
    img_out = np.zeros(img.shape)

    for linha in range(int(JANELA_EROSAO/2),rows-int(JANELA_EROSAO/2)):
        for coluna in range(int(JANELA_EROSAO/2),cols-int(JANELA_EROSAO/2)):
            and_ = True
            for y in range(linha-int(JANELA_EROSAO/2),linha+int(JANELA_EROSAO/2)+1):
                for x in range(coluna-int(JANELA_EROSAO/2),coluna+int(JANELA_EROSAO/2)+1):
                    if(img[y][x] == 0 ):
                        and_ = False
                        break
                if(not and_):
                    break

            img_out[linha][coluna] = and_

    return img_out

def Dilata(img):
    print("\t\tDilata")
    rows, cols = img.shape
    img_out = np.zeros(img.shape)

    for linha in range(int(JANELA_DILATACAO/2),rows-int(JANELA_DILATACAO/2)):
        for coluna in range(int(JANELA_DILATACAO/2),cols-int(JANELA_DILATACAO/2)):
            or_ = False
            for y in range(linha-int(JANELA_DILATACAO/2),linha+int(JANELA_DILATACAO/2)+1):
                for x in range(coluna-int(JANELA_DILATACAO/2),coluna+int(JANELA_DILATACAO/2)+1):
                    if(img[y][x] == 1 ):
                        or_ = True
                        break
                if(or_):
                    break

            img_out[linha][coluna] = or_

    return img_out

def defineSigma(w, h):
    #Slide HDR, pag 45
    return int(min(w,h)/24)


#Binarizacao com Treshhold local
def LimirarizacaoAdaptativa(img):
    
    rows, cols = img.shape
    sigma = defineSigma(cols, rows)
    print("\tSigma: ", sigma)
    blur = cv2.GaussianBlur(img, (0,0), sigma)  
    
    img_bin = np.where( img-blur > THRESHOLD, 1, 0)

    # img_bin = np.zeros(img.shape)
    # for linha in range(rows):
    #     for coluna in range(cols):
    #         img_bin[linha][coluna] = 0
    #         if((img[linha][coluna] - blur[linha][coluna]) > THRESHOLD):
    #             img_bin[linha][coluna] = 1

    img_erosao = Erosao(img_bin)
    img_dilata = Dilata(img_erosao)
    img_out = np.concatenate((img_bin,img_erosao, img_dilata), axis=1)
    return img_out

def contaArroz(imagem):
    #gray_image = cv2.cvtColor(integral, cv2.COLOR_BGR2GRAY)
    mask = LimirarizacaoAdaptativa(imagem)

    return mask

def main():
    for img in INPUT_IMAGE:
        print("imagem:", img)
        imagem = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # Normalizando com float
        imagem = imagem.astype (np.float32) / 255
        
        start_time = timeit.default_timer ()
        img_out = contaArroz(imagem)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        
        vert = np.concatenate((imagem, img_out), axis=1)
        vert = cv2.resize(vert, (int(vert.shape[1]/2), int(vert.shape[0]/2)))
        cv2.imshow (img, vert)
        cv2.imwrite (img+'_out.png', vert*255)
    
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()

