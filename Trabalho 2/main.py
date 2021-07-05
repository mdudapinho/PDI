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

INPUT_IMAGE =  r'.\\Exemplos\\b01 - Original.bmp'  #documento-3mp
JANELA_H = 3
JANELA_W = 3
MODE = 2 # 0:INGENUO, 1:SEPRAVEL, 2:INTEGRAL, 3:TODOS
LIMITE = 0.004 # Indice que limita a diferenca por pixel no comparador
 
#===============================================================================

def comparador(img_1,img_2,rows, cols, integral = False):
    sum = 0

    #Se nao for do algoritmo imagens integrais, descarta as bordas
    if(not integral):
        #desconta as margens
        r = rows-int(JANELA_H/2)*2
        c = cols-int(JANELA_W/2)*2
        
        # Compara pixel a pixel
        for linha in range(int(JANELA_H/2),rows-int(JANELA_H/2)):        
            for coluna in range(int(JANELA_W/2),cols-int(JANELA_W/2)):
                dif = img_1[linha][coluna] - img_2[linha][coluna] 
                if(dif < LIMITE):
                    sum += 1

        sum = sum/(r*c)
        print("\t\tAs imagens sao ", round(sum*100, 2), "% parecidas")
        return

    # Se imagensIntegrais, compara a imagem inteira sem excluir o 
    # tamanho das janelas das bordas
    else:
        # Compara pixel a pixel
        for linha in range(rows):        
            for coluna in range(cols):
                dif = img_1[linha][coluna] - img_2[linha][coluna]
                if(dif < LIMITE):
                    sum += 1

        sum = sum / (rows * cols)
        print("\t\tAs imagens sao ", round(sum*100, 2), "% parecidas")
        return

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
    
    ## Percorre na linha, acumulando os valores a esquerda 
    ## Primeira coluna da imagem ja esta formada
    for linha in range(rows):        
        for coluna in range(1,cols):
            # Buffer = Intensidade no pixel + Pixel a esquerda
            buffer[linha][coluna] = img_aux[linha][coluna] + buffer[linha][coluna-1]

    ## Percorro por linha
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

def integral(img):

    rows, cols, channels = img.shape
    
    # Imagem integral
    buffer = createBuffer(rows, cols, img)
    
    #img_out com zeros
    img_out = np.zeros((rows, cols))

    # Janela deslizante
    for linha in range(rows):        
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

    # Mantem uma copia colorida para desenhar a saida
    img_color = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Calcula filtro com OpenCV
    img_out_opencv = opencvblur(img)
    
    rows_, cols_, channels = img.shape

    if(MODE == 0  or MODE == 3):
        # INGENUO
        img_ingenuo = img
        cv2.imshow ('img entrada ingenuo',img_ingenuo)
        cv2.waitKey ()
        start_time = timeit.default_timer ()
        print("Iniciou ingenuo")
        img_out_ingenuo = ingenuo(img_ingenuo)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        print("\tcomparador (openCV e ingenuo: ")
        comparador(img_out_ingenuo,img_out_opencv,rows_, cols_)
        cv2.imwrite ('out_ingenuo.png', img_out_ingenuo*255)
        cv2.imshow ('saida ingenuo',img_out_ingenuo)

    if(MODE == 1 or MODE == 3):
        # SEPARAVEL
        img_separavel = img
        cv2.imshow ('img entrada separavel',img_separavel)
        cv2.waitKey ()
        start_time = timeit.default_timer ()
        print("Iniciou separavel")
        img_out_separavel = separavel(img_separavel)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        print("\tcomparador (openCV e sepravel: ")
        comparador(img_out_separavel,img_out_opencv,rows_, cols_)
        cv2.imwrite ('out_separavel.png', img_out_separavel*255)
        cv2.imshow ('saida separavel', img_out_separavel)
        
    if(MODE == 2 or MODE == 3):
        # INTEGRAL
        img_integral = img
        cv2.imshow ('img entrada integral',img_integral)
        cv2.waitKey ()
        start_time = timeit.default_timer ()
        print("Iniciou Integral")
        img_out_integral = integral(img_integral)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        print("\tcomparador (openCV e integral): ")
        comparador(img_out_integral,img_out_opencv,rows_, cols_,True)
        cv2.imwrite ('out_integral.png', img_out_integral*255)
        cv2.imshow ('saida integral',img_out_integral)

    # Comparacao com filtro do openCV
    cv2.imwrite ('out_opencv.png', img_out_opencv*255)
    cv2.imshow ('saida opencv',img_out_opencv)

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()