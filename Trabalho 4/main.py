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

SIGMA = 100

JANELA_H = 157
JANELA_W = 157

M_H = 5
M_W = 5

LIMIAR = 0.2
#===============================================================================

def createBuffer(rows, cols, img_aux):
    buffer = np.zeros(img_aux.shape)

    ## Percorre na linha, acumulando os valores a esquerda
    for linha in range(rows):
        for coluna in range(cols):
            ## Primeira coluna da imagem copia o valor
            if(linha == 0 ):
                buffer[linha][coluna] = img_aux[linha][coluna]

            # Buffer = Intensidade no pixel + Pixel a esquerda
            buffer[linha][coluna] = img_aux[linha][coluna] + buffer[linha][coluna-1]

    ## Percorre por linha
    ## Acumulando o valor que esta em cima do pixel no buffer
    ## Completa a imagem integral
    for linha in range(1,rows):
        for coluna in range(cols):
            # Buffer = Intensidade no buffer + Valor do buffer em cima
            buffer[linha][coluna] = buffer[linha][coluna] + buffer[linha-1][coluna]

    return buffer

def calculaMedia(linha, coluna, buffer_, rows, cols):
    media = 0

    superior = linha - int(JANELA_H/2) - 1
    if(superior < 0 ):
        superior = -1

    inferior = linha + int(JANELA_H/2)
    if (inferior > rows - 1 ):
        inferior = rows - 1

    esquerda = coluna - int(JANELA_W/2) - 1
    if(esquerda < 0 ):
        esquerda = -1

    direita = coluna + int(JANELA_W/2)
    if (direita > cols - 1 ):
        direita = cols - 1

    # Estabelece os limites da janela
    # Quando nas bordas, a janela e menor
    janela_w = direita - esquerda
    janela_h = inferior - superior

    # Definindo os 4 pontos
    soma = buffer_[inferior][direita]
    if(superior >= 0 ):
        soma = soma - buffer_[superior][direita]
    if(esquerda >= 0 ):
        soma = soma - buffer_[inferior][esquerda]
    if(superior >= 0 and esquerda >= 0 ):
        soma = soma + buffer_[superior][esquerda]

    media = soma / (janela_w * janela_h)

    return media

def Integral(img):

    rows, cols = img.shape

    # Imagem integral
    buffer = createBuffer(rows, cols, img)

    img_out = np.zeros(img.shape)

    # Janela deslizante
    for linha in range(rows):
        for coluna in range(cols):
            img_out[linha][coluna] = calculaMedia(linha, coluna, buffer, rows, cols)

    return img_out

def Erosao(img):
    print("\tErode")

    rows, cols = img.shape
    img_out = np.zeros(img.shape)

    for linha in range(int(M_H/2),rows-int(M_H/2)):
        for coluna in range(int(M_W/2),cols-int(M_W/2)):
            and_ = True
            for y in range(linha-int(M_H/2),linha+int(M_H/2)+1):
                for x in range(coluna-int(M_W/2),coluna+int(M_W/2)+1):
                    if(img[y][x] == 0 ):
                        and_ = False

            img_out[linha][coluna] = and_

    return img_out

def Dilata(img):
    print("\t\tDilata")
    rows, cols = img.shape
    img_out = np.zeros(img.shape)

    for linha in range(int(M_H/2),rows-int(M_H/2)):
        for coluna in range(int(M_W/2),cols-int(M_W/2)):
            or_ = False
            for y in range(linha-int(M_H/2),linha+int(M_H/2)+1):
                for x in range(coluna-int(M_W/2),coluna+int(M_W/2)+1):
                    if(img[y][x] == 1 ):
                        or_ = True

            img_out[linha][coluna] = or_

    return img_out
#Binarizacao com Treshhold local
def LimirarizacaoAdaptativa(img):
    blur = cv2.GaussianBlur(img, (0,0), SIGMA)  #Integral(img)
    rows, cols = img.shape
    img_bin = np.zeros(img.shape)
    for linha in range(rows):
        for coluna in range(cols):
            img_bin[linha][coluna] = 0
            if(abs(img[linha][coluna] - blur[linha][coluna]) > LIMIAR):
                img_bin[linha][coluna] = 1

    img_erosao = Dilata(Erosao(img_bin))
    img_out = np.concatenate((img_bin,img_erosao), axis=1)
    return img_out

def contaArroz(imagem):
    #gray_image = cv2.cvtColor(integral, cv2.COLOR_BGR2GRAY)
    mask = LimirarizacaoAdaptativa(imagem)

    return mask

def main():
    a = 1
    for img in INPUT_IMAGE:
        print("imagem:", img)
        imagem = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # Normalizando com float
        imagem = imagem.astype (np.float32) / 255
        img_out = contaArroz(imagem)
        vert = np.concatenate((imagem, img_out), axis=1)
        vert = cv2.resize(vert, (int(vert.shape[1]/2), int(vert.shape[0]/2)))
        cv2.imshow (img, vert)
        a+=1
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()

