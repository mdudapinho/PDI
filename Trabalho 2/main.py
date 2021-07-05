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

#===============================================================================

def comparador(img_1,img_2):
    
    rows, cols, channels = img_1.shape
    img_ = img_1

    print (img_2)

    for linha in range(rows):
        for coluna in range(cols):
            img_[linha][coluna] = img_1[linha][coluna] - img_2[linha][coluna] 

    return img_

def opencvblur(img):
    # blur (img, largura, altura)
    img_out = cv2.blur(img,(JANELA_W,JANELA_H))
    return img_out

def ingenuo(img):
    
    rows, cols, channels = img.shape
    img_out = img

    for linha in range(int(JANELA_H/2),rows-int(JANELA_H/2)):        
        print("Linhas:",linha)
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
        print("Linhas:",linha)
        for coluna in range(int(JANELA_W/2),cols-int(JANELA_W/2)):
            soma = 0
            # iteracao na horizontal
            for x in range(coluna-int(JANELA_W/2),coluna+int(JANELA_W/2)+1):
                soma += img[linha][x]
            buffer[linha][coluna] = soma / JANELA_W

    # VERTICAL
    for linha in range(int(JANELA_H/2),rows-int(JANELA_H/2)):        
        print("Linhas:",linha)
        for coluna in range(int(JANELA_W/2),cols-int(JANELA_W/2)):
            soma = 0
            # iteracao na vertical
            for y in range(linha-int(JANELA_H/2),linha+int(JANELA_H/2)+1):
                soma += buffer[y][coluna]
            img_out[linha][coluna] = soma / JANELA_H

    return img_out

def calculaMedia(linha, coluna, buffer_, rows, cols):
    ponto_superior = linha - int(JANELA_H/2) - 1
            
    ponto_inferior = linha + int(JANELA_H/2)
    if (ponto_inferior > rows - 1 ):
        ponto_inferior = rows - 1

    ponto_esquerda = coluna - int(JANELA_W/2) - 1 
    
    ponto_direita = coluna + int(JANELA_W/2) 
    if (ponto_direita > cols - 1 ):
        ponto_direita = cols - 1

    janela_w = ponto_direita + 1
    if(ponto_esquerda >= 0 ):
        janela_w = ponto_direita - ponto_esquerda

    janela_h = ponto_inferior + 1
    if(ponto_superior >= 0 ):
        janela_h = ponto_inferior - ponto_superior

    # Definindo os 4 pontos
    print("-------")
    print("coluna: ", coluna)
    print("ponto_superior: ", ponto_superior)
    print("ponto_inferior: ", ponto_inferior)
    print("ponto_esquerda: ", ponto_esquerda)
    print("ponto_direita: ", ponto_direita)
    print("janela_w: ", janela_w)
    print("janela_h: ", janela_h)

    print("soma: ", buffer_[ponto_inferior][ponto_direita])
    soma = buffer_[ponto_inferior][ponto_direita]
    if(ponto_superior >= 0 ):
        soma = soma - buffer_[ponto_superior][ponto_direita]
        print("sup: diminui ", buffer_[ponto_superior][ponto_direita], " -> soma: ", soma)
    if(ponto_esquerda >= 0 ):
        soma = soma - buffer_[ponto_inferior][ponto_esquerda]
        print("esq: diminui ", buffer_[ponto_inferior][ponto_esquerda], " -> soma: ", soma)
    if(ponto_superior >= 0 and ponto_esquerda >= 0 ):
        soma = soma + buffer_[ponto_superior][ponto_esquerda]
        print("dois: soma ", buffer_[ponto_superior][ponto_esquerda], " -> soma: ", soma)
    
    media = soma / (janela_w * janela_h)
    if(media != 1):
        print("\tsoma: ", soma)
        print("\tmedia: ", media)
    
    return media

def integral(img):

    img_aux = np.where( img > 0, 1, 1)
    rows, cols, channels = img.shape
    img_out = img_aux
    buffer = img_aux
    
    # Imagem integral
    ## Percorro a coluna 
    ## Primeira coluna da imagem ja esta formada
    
    for linha in range(rows):        
        print("Linhas:",linha)
        for coluna in range(1,cols):
            buffer[linha][coluna] = img_aux[linha][coluna] + buffer[linha][coluna-1]

    ## Percorro por linha 
    for linha in range(1,rows):        
        print("Linhas:",linha)
        for coluna in range(cols):
            buffer[linha][coluna] = buffer[linha][coluna] + buffer[linha-1][coluna]
    
    # for linha in range(rows):        
    #     for coluna in range(cols):    
    #         if(buffer[linha][coluna] != (linha+1)*(coluna+1)):
    #             print("buffer[", linha, "][", coluna, "]: ", buffer[linha][coluna])
    
    # Janela deslizante
    for linha in range(rows):        
        print("Linhas:",linha)
        # for y in range(rows):        
        #     for x in range(cols):    
        #         if(buffer[y][x] != (y+1)*(x+1)):
        #             print("buffer[", y, "][", x, "]: ", buffer[y][x])
        for coluna in range(cols):           
            img_out[linha][coluna] = calculaMedia(linha, coluna, buffer, rows, cols)

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
   
    # Colocar funcoes aqui dentro
    # print("Iniciou ingenuo")
    # img_out_ingenuo = ingenuo(img)

    # print("Iniciou separavel")
    # img_out_separavel = separavel(img)

    img_out_integral = integral(img)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))

    #img_comparada = comparador(img_out_separavel,img_opencv)
    cv2.imwrite ('out.png', img_out_integral*255)
    cv2.imshow ('saida borrada ', img_out_integral)
    cv2.imshow ('saida opencv',img_opencv)
    # cv2.imshow ('comparador',img_comparada)
    cv2.imwrite ('out.png', img_out_integral*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()