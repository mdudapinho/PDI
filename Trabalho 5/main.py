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

#===============================================================================

INPUT_IMAGE = [r"./img/1.bmp"]#, r"./img/1.bmp" ]

#===============================================================================
def defH(h_, lower_green, lower_center_green, upper_center_green, upper_green):
    
    h = 1
    if( h_ > lower_green and h_ < lower_center_green ):
        h = 1 - (h_-lower_green) / (lower_center_green - lower_green)
    elif (h_ >= lower_center_green and h_ <= upper_center_green):
        h = 0
    elif(h_ > upper_center_green and h_ < upper_green):
        h = (h_-upper_center_green) * (upper_green - upper_center_green)
    return h

def createMask(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = [40, 40,40]
    lower_center_green = [50, 150,150]
    upper_center_green = [60, 200,200]
    upper_green = [70,255,255]

    #mask = cv2.inRange(imgHSV, lower_green, upper_green)
    mask = np.zeros(img.shape)
    rows, cols, _ = imgHSV.shape

    for linha in range(rows):
        for coluna in range(cols):
            h = defH(imgHSV[linha][coluna][0], lower_green[0], lower_center_green[0], upper_center_green[0], upper_green[0])
            s = defH(imgHSV[linha][coluna][1], lower_green[1], lower_center_green[1], upper_center_green[1], upper_green[1])
            v = defH(imgHSV[linha][coluna][2], lower_green[2], lower_center_green[2], upper_center_green[2], upper_green[2])
            
            mask[linha][coluna] = (h*s*v)/1

    
    res = mask#cv2.bitwise_and(img,img, mask= mask)

    return res

def main():
    for img in INPUT_IMAGE:
        print("imagem:", img)
        imagem = cv2.imread(img)
        if imagem is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # Filtro de verde
        greenPass = createMask(imagem)
        imagem = imagem.astype (np.float32) / 255

        start_time = timeit.default_timer ()
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))

        # Concatena imagens na tela para visualização
        #vert = np.concatenate((imagem), axis=1)
        imagem = cv2.resize(imagem, (int(imagem.shape[1]/2), int(imagem.shape[0]/2)))
        greenPass = cv2.resize(greenPass, (int(greenPass.shape[1]/2), int(greenPass.shape[0]/2)))
        cv2.imshow (img, imagem)
        cv2.imshow ("mask"+img, greenPass)
        #cv2.imwrite (img+'_out.png', vert*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
