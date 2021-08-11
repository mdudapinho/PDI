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
import statistics

#===============================================================================

INPUT_IMAGE =  [r"./60.bmp", r"./82.bmp", r"./114.bmp", r"./150.bmp", r"./205.bmp"]
#INPUT_IMAGE =  [r"./150.bmp"]

JANELA_EROSAO = 5
JANELA_DILATACAO = 5
LIMITE_ACEITACAO = 1.3 #se o blob for maior que [LIMITE_ACEITACAO*mediana dos blobs], tem mais de um junto
MAX_ARROZ = 1.125 #Margem para definir o tamanho máximo de um arroz dentro de um blob (MAX_ARROZ*mediana)
THRESHOLD = 0.2

SAVE_IMAGES = True

#===============================================================================

#Erosao da imagem, com a janela estabelecida em Macro
def Erosao(img):
    print("\tErode")

    img = img.astype('uint8')
    kernel = np.ones((JANELA_EROSAO,JANELA_EROSAO),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)

    return erosion

#Dilatacao da imagem, com a janela estabelecida em Macro
def Dilata(img):
    print("\tDilata")

    img = img.astype('uint8')
    kernel = np.ones((JANELA_DILATACAO,JANELA_DILATACAO),np.uint8)
    erosion = cv2.dilate(img,kernel,iterations = 1)

    return erosion

# Funcao de inundacao (usada no primeiro trabalho)
def FindBlob (label,img, y0,x0, blob):
    # Marca o arroz com o valor dele
    img[y0][x0] = label
    blob.append({'x': x0, 'y': y0, 'label': label})

    # Tem vizinho para direita
    if (x0+1 < img.shape[1] and img[y0][x0+1] == -1 ):
        blob = FindBlob(label,img,y0,x0+1, blob)

    # Tem vizinho para esquerda
    if (x0 > 0  and img[y0][x0-1] == -1):
        blob = FindBlob(label,img,y0,x0-1, blob)

    # Tem vizinho pra cima
    if (y0-1 >= 0  and img[y0-1][x0] == -1 ):
        blob = FindBlob(label,img,y0-1,x0, blob)

    # Tem vizinho pra baixo
    if (y0+1 < img.shape[0] and img[y0+1][x0] == -1 ):
        blob = FindBlob(label,img,y0+1,x0, blob)

    return blob

def rotula (img):
    print("\t\t\trotulando")
    rows, cols = img.shape

    label = 1
    # Difere o pixel marcado com 1, para começar o label como 1.
    img_ = np.zeros(img.shape)
    img_ = np.where( img == 1 , -1, 0)

    # Define uma lista de espacos identificados como arroz
    blobs =[]

    # Procura na imagem pixels brancos
    for linha in range(rows):
        for coluna in range(cols):
            # Encontra pixel capaz de ser arroz
            if (img_[linha][coluna] == -1):
                blob = FindBlob(label, img_,linha,coluna,[])
                blobs.append(blob)
                label = label + 1

    return blobs

def countBlobs(blobs):
    lens = []
    for blob in blobs:
        lens.append(len(blob))

    mediana = statistics.median(lens)

    blob_counter = 0
    for blob in blobs:
        blob_counter += 1
        if(len(blob)/LIMITE_ACEITACAO > mediana):
            blob_counter += int(len(blob)/(mediana*MAX_ARROZ))

    return blob_counter

def defineSigma(w, h):
    #Slide HDR, pag 45
    return int(min(w,h)/24)

#Binarizacao com Treshhold local
def LimirarizacaoAdaptativa(img):

    rows, cols = img.shape
    sigma = defineSigma(cols, rows)
    blur = cv2.GaussianBlur(img, (0,0), sigma)

    img_bin = np.where( img-blur > THRESHOLD, 1, 0)

    return img_bin

def contaArroz(imagem):
    #gray_image = cv2.cvtColor(integral, cv2.COLOR_BGR2GRAY)
    img_limiarizada = LimirarizacaoAdaptativa(imagem)

    img_erodida = Erosao(img_limiarizada)
    img_dilatada = Dilata(img_erodida)
    img_out = np.concatenate((img_limiarizada,img_erodida, img_dilatada), axis=1)

    blobs = rotula(img_dilatada)

    arroz_counter = countBlobs(blobs)
    return img_out, arroz_counter

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
        img_out, arroz_counter = contaArroz(imagem)
        print("\tencontrou: ", arroz_counter)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        vert = np.concatenate((imagem, img_out), axis=1)
        vert = cv2.resize(vert, (int(vert.shape[1]/2), int(vert.shape[0]/2)))
        cv2.imshow (img, vert)
        if(SAVE_IMAGES):
            cv2.imwrite (img+'_out.png', vert*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
