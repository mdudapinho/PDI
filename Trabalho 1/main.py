#===============================================================================
# Exemplo: segmentacao de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnologica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'  #documento-3mp

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.6     #0.4
ALTURA_MIN = 10     #5
LARGURA_MIN = 10    #5
N_PIXELS_MIN = 10

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, da para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    rows, cols, channels = img.shape
    img_out = np.where( img > threshold, 1, 0)

    return img_out 
#-------------------------------------------------------------------------------
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
    if (y0-1 > 0  and img[y0-1][x0] == -1 ):
        blob = FindBlob(label,img,y0-1,x0, blob)

    # Tem vizinho pra baixo
    if (y0+1 < img.shape[0] and img[y0+1][x0] == -1 ):
        blob = FindBlob(label,img,y0+1,x0, blob)

    return blob

def CheckBlob(blob, largura_min, altura_min, n_pixels_min):
    
    n_pixels = len(blob)
    if(n_pixels < n_pixels_min):
        return False
    
    topo = blob[0]['y']
    baixo = blob[0]['y']
    esquerda = blob[0]['x']
    direita = blob[0]['x']

    for pos in blob:
        if(topo > pos['y']):
            topo = pos['y']
        if(baixo < pos['y']):
            baixo = pos['y']
        if(direita < pos['x']):
            direita = pos['x']
        if(esquerda > pos['x']):
            esquerda = pos['x']

    if((baixo - topo) < altura_min):
        return False
    if((direita - esquerda) < largura_min):
        return False
    
    res = {
        "label": blob[0]['label'],
        "n_pixels": n_pixels,
        'T': topo, 
        'L': esquerda,
        'B':baixo, 
        'R': direita,
    }
    print(res)
    return res

def rotula (img, largura_min, altura_min, n_pixels_min, img_out):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    
    rows, cols, channels = img.shape
    label = 1
    img_ = np.where( img == 1 , -1, 0)
    blobs =[]
    for linha in range(rows):
        for coluna in range(cols):
            # Tem arroz aqui
            if (img_[linha][coluna] == -1):
                blob = FindBlob(label, img_,linha,coluna,[])
                checkedBlob = CheckBlob(blob, largura_min, altura_min, n_pixels_min)
               
                if(checkedBlob):
                    blobs.append(checkedBlob)
                    label = label + 1
    return blobs
#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    # Muda o numero de linhas, colunas e canais 
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    # Normalizando com float
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    # cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN, img_out)
    
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
