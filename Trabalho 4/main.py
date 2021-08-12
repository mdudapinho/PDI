#===============================================================================
# Trabalho 4
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

""" 
Imagens para teste
"""
INPUT_IMAGE =  [r"./60.bmp", r"./82.bmp", r"./114.bmp", r"./150.bmp", r"./205.bmp"]

""" 
Constantes de morfologia.
"""
JANELA_EROSAO = 5
JANELA_DILATACAO = 5

""" 
Constantes de cálculo de arroz.

- LIMITE_ACEITACAO
    Se o blob for maior que [LIMITE_ACEITACAO*mediana dos blobs], existe mais de um arroz junto neste blob.
    Exemplo: Se LIMITE_ACEITACAO for igual a 1.5, os blobs 1.5 vezes maiores que a mediana são considerados 
    como uma aglomeraçao, desta forma, tendo mais de 1 arroz no grupo.

- MAX_ARROZ
    Define o tamanho máximo de um arroz dentro de um blob sinalizado como aglomeração.
    Esse limite é dado pelo cálculo MAX_ARROZ*mediana.

    # Isso pois em alguns blobs nem todos os arroz eram representados após a morfologia.
    # Em alguns casos, blobs muito aglomerados podiam conter mais ou menos arroz do que na foto original.
    # Deste jeito, tivemos que adicionar uma margem de limite para considerar os espaços não preenchidos.

    Exemplo: Se MAX_ARROZ for igual a 1.5, o arroz dentro do blob terá no máximo 1.5 vezes o tamanho da mediana.

- THRESHOLD
    Define o limite para a limiarização adaptativa da imagem. A conta é feita utilizando a imagem original menos a imagem borrada, 
    após isso o np.where precisa do THRESHOLD para verificar a binarização.

- SIGMA
    Define o tamanho da janela do cv2.GaussianBlur. 
"""
LIMITE_ACEITACAO = 1.3
MAX_ARROZ = 1.125 
THRESHOLD = 0.2
SIGMA = 32

#===============================================================================

"""
    Erosao da imagem com a janela estabelecida na macro JANELA_EROSAO
"""
def Erosao(img):
    print("\tErode")

    img = img.astype('uint8')
    kernel = np.ones((JANELA_EROSAO,JANELA_EROSAO),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)

    return erosion

""" 
    Dilatacao da imagem, com a janela estabelecida na Macro JANELA_DILATACAO
"""
def Dilata(img):
    print("\tDilata")

    img = img.astype('uint8')
    kernel = np.ones((JANELA_DILATACAO,JANELA_DILATACAO),np.uint8)
    erosion = cv2.dilate(img,kernel,iterations = 1)

    return erosion

""" 
    Função de inundacao (usada no trabalho 1)
"""
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

""" 
    Função de identificacao dos blobs (adaptada do trabalho 1)
"""
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

""" 
    Função de contagem de arroz dentro dos blobs.
    Essa função usa a mediana e os limites estabelecidos nas macros LIMITE_ACEITACAO e MAX_ARROZ
"""
def countBlobs(blobs):
    lens = []

    # Obtem o tamanho de cada blob obtido
    for blob in blobs:
        lens.append(len(blob))

    # Encontra-se a mediana do tamanho dos blobs
    mediana = statistics.median(lens)

    blob_counter = 0
    for blob in blobs:
        # Achou 1 arroz, pelo menos
        blob_counter += 1
        # Verifica se o blob é maior do que o esperado com o LIMITE_ACEITACAO
        if(len(blob)/LIMITE_ACEITACAO > mediana):
            # Se o blob for maior, divide-se o blob em partes equivalente a mediana*MAX_ARROZ. 
            blob_counter += int(len(blob)/(mediana*MAX_ARROZ))

    return blob_counter

""" 
    Binarizacao com Treshhold local
    Nesta função a imagem principal é borrada com a macro SIGMA.
    A binarização ocorre quando um pixel na imagem original menos um pixel na imagem borrada for 
    maior que o THRESHOLD. Essa binarização enalteceu as formas para a realização da abertura, por isso
    não foi utilizada a cv2.adaptiveThreshold (Função do opencv).
"""
def LimiarizacaoAdaptativa(img):

    blur = cv2.GaussianBlur(img, (0,0), SIGMA)
    img_bin = np.where( img-blur > THRESHOLD, 1, 0)

    return img_bin

""" 
    Contagem de arroz.
    Nesta função ocorrem as principais funções de tratamento.
    A limiarização adaptativa binariza a imagem com treshold local.
    A abertura (Erosão + Dilatação) da imagem é feita para remoção de ruidos de fundo dada a binarização.
    Após isso, segue para a contagem de blobs e, com a definição dos blobs, para a contagem de arroz.
"""
def contaArroz(imagem):
    img_limiarizada = LimiarizacaoAdaptativa(imagem)

    # Realização da abertura da imagem para remoção de ruidos de fundo.
    img_erodida = Erosao(img_limiarizada)
    img_dilatada = Dilata(img_erodida)

    # Concatena imagem para visualização na função principal
    img_out = np.concatenate((img_limiarizada, img_erodida, img_dilatada), axis=1)

    # Retorna o número de blobs existentes
    blobs = rotula(img_dilatada)

    # Conta arroz em cada blob.
    arroz_counter = countBlobs(blobs)

    return img_out, arroz_counter

""" 
    Função principal.
    Inicia a temporização e as funções principais de contagem do arroz. 
"""
def main():
    for img in INPUT_IMAGE:
        print("imagem:", img)
        imagem = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # Normalizando com float
        imagem = imagem.astype (np.float32) / 255

        # Inicio de contagem de arroz
        start_time = timeit.default_timer ()
        img_out, arroz_counter = contaArroz(imagem)
        print("\tencontrou: ", arroz_counter)
        # Fim
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        
        # Concatena imagens na tela para visualização
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
