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

INPUT_IMAGE =  'a01.bmp'  #documento-3mp
JANELA_H = 5
JANELA_W = 5
MODE = 0 # 0:INGENUO, 1:SEPRAVEL, 2:INTEGRAL, 3:TODOS
#===============================================================================

def comparador(img_1,img_2,rows, cols):
    
    sum = 0

    for linha in range(rows):
        for coluna in range(cols):
            if(img_1[linha][coluna] == img_2[linha][coluna] ):
                sum += 1
    sum = sum/(rows*cols)
    print("As imagens sao ", sum*100, "% parecidas")


def opencvblur(img):
    # blur (img, largura, altura)
    img_out = cv2.blur(img,(JANELA_W,JANELA_H))
    return img_out

def ingenuo(img):
    
    rows, cols, channels = img.shape
    img_out = img

    for linha in range(int(JANELA_H/2),rows-int(JANELA_H/2)):        
        for coluna in range(int(JANELA_W/2),cols-int(JANELA_W/2)):
            soma = 0
            for y in range(linha-int(JANELA_H/2),linha+int(JANELA_H/2)+1):
                for x in range(coluna-int(JANELA_W/2),coluna+int(JANELA_W/2)+1):
                    soma += img[y][x]

            img_out[linha][coluna] = soma/(JANELA_H*JANELA_W)

    return img_out

def separavel(img):
    
    rows, cols, channels = img.shape
    img_out = img
    buffer = img
    
    # HORIZONTAL
    for linha in range(int(JANELA_H/2),rows-int(JANELA_H/2)):        
        for coluna in range(int(JANELA_W/2),cols-int(JANELA_W/2)):
            soma = 0
            # iteracao na horizontal
            for x in range(coluna-int(JANELA_W/2),coluna+int(JANELA_W/2)+1):
                soma += img[linha][x]
            buffer[linha][coluna] = soma / JANELA_W

    # VERTICAL
    for linha in range(int(JANELA_H/2),rows-int(JANELA_H/2)):        
        for coluna in range(int(JANELA_W/2),cols-int(JANELA_W/2)):
            soma = 0
            # iteracao na vertical
            for y in range(linha-int(JANELA_H/2),linha+int(JANELA_H/2)+1):
                soma += buffer[y][coluna]
            img_out[linha][coluna] = soma / JANELA_H

    return img_out

def createBuffer(rows, cols, img_aux):
    buffer = img_aux
    
    ## Percorro a coluna 
    ## Primeira coluna da imagem ja esta formada
    for linha in range(rows):        
        for coluna in range(1,cols):
            buffer[linha][coluna] = img_aux[linha][coluna] + buffer[linha][coluna-1]

    ## Percorro por linha 
    for linha in range(1,rows):        
        for coluna in range(cols):
            buffer[linha][coluna] = buffer[linha][coluna] + buffer[linha-1][coluna]
    
    return buffer


def calculaMedia(linha, coluna, buffer_, rows, cols):
    media = 0

    ponto_superior = linha - int(JANELA_H/2) - 1
    if(ponto_superior < 0 ):
        ponto_superior = -1
            
    ponto_inferior = linha + int(JANELA_H/2)
    if (ponto_inferior > rows - 1 ):
        ponto_inferior = rows - 1

    ponto_esquerda = coluna - int(JANELA_W/2) - 1 
    if(ponto_esquerda < 0 ):
        ponto_esquerda = -1
    
    ponto_direita = coluna + int(JANELA_W/2) 
    if (ponto_direita > cols - 1 ):
        ponto_direita = cols - 1

    janela_w = ponto_direita - ponto_esquerda
    janela_h = ponto_inferior - ponto_superior
    
    # Definindo os 4 pontos
    soma = buffer_[ponto_inferior][ponto_direita]
    if(ponto_superior >= 0 ):
        soma = soma - buffer_[ponto_superior][ponto_direita]
    if(ponto_esquerda >= 0 ):
        soma = soma - buffer_[ponto_inferior][ponto_esquerda]
    if(ponto_superior >= 0 and ponto_esquerda >= 0 ):
        soma = soma + buffer_[ponto_superior][ponto_esquerda]
    
    media = soma / (janela_w * janela_h)
    return media

def integral(img):

    rows, cols, channels = img.shape
    
    # Imagem integral
    buffer = createBuffer(rows, cols, img)
    
    #img_out = buffer
    img_out = np.zeros((rows, cols))

    # Janela deslizante
    for linha in range(rows):        
        for coluna in range(cols):           
            a = calculaMedia(linha, coluna, buffer, rows, cols)
            img_out[linha][coluna] = a
            
    return img_out

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_color = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # cv2.imshow ('01 - binarizada', img)
    # cv2.imwrite ('01 - binarizada.png', img*255)

    img_opencv = opencvblur(img)
    
    start_time = timeit.default_timer ()
   
    rows_, cols_, channels = img.shape
    if(MODE == 0  or MODE == 3):
        # INGENUO
        print("Iniciou ingenuo")
        img_out_ingenuo = ingenuo(img)
        print ('Tempo: %f' % (timeit.default_timer () - start_time))
        print("comparador (openCV e ingenuo: ")
        comparador(img_out_ingenuo,img_opencv,rows_, cols_)

    if(MODE == 1 or MODE == 3):
        # SEPARAVEL
        #start_time = timeit.default_timer ()
        print("Iniciou separavel")
        img_out_separavel = separavel(img)
        print ('Tempo: %f' % (timeit.default_timer () - start_time))
        print("comparador (openCV e sepravel: ")
        comparador(img_out_separavel,img_opencv,rows_, cols_)
        
    if(MODE == 2 or MODE == 3):
        # INTEGRAL
        #start_time = timeit.default_timer ()
        print("Iniciou Integral")
        img_out_integral = integral(img)
        print ('Tempo: %f' % (timeit.default_timer () - start_time))
        print("comparador (openCV e integral: ")
        comparador(img_out_integral,img_opencv,rows_, cols_)
        
    #img_comparada = comparador(img_out_separavel,img_opencv)

    cv2.imwrite ('out_opencv.png', img_opencv*255)
    cv2.imshow ('saida opencv',img_opencv)

    if(MODE == 0  or MODE == 3):
        cv2.imwrite ('out_ingenuo.png', img_out_ingenuo*255)
        cv2.imshow ('saida ingenuo',img_out_ingenuo)
    if(MODE == 1 or MODE == 3):
        cv2.imwrite ('out_separavel.png', img_out_separavel*255)
        cv2.imshow ('saida separavel', img_out_separavel)
    if(MODE == 2 or MODE == 3):
        cv2.imwrite ('out_integral.png', img_out_integral*255)
        cv2.imshow ('saida integral',img_out_integral)


    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()