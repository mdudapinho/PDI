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

INPUT_IMAGE = [r"./img/0.BMP", r"./img/1.bmp" ]

#===============================================================================
def createMask(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40,40])
    upper_green = np.array([70,255,255])

    mask = cv2.inRange(imgHSV, lower_green, upper_green)
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
        cv2.imshow (img+"mask", greenPass)
        #cv2.imwrite (img+'_out.png', vert*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
