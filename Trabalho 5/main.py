#===============================================================================
# Trabalho 5
#-------------------------------------------------------------------------------
# Equipe:
# Lucas Ricardo Marques de Souza
# Maria Eduarda Pinho
#===============================================================================

import numpy as np
import timeit
import sys
import cv2

#===============================================================================

INPUT_IMAGE = [r"./img/0.bmp",r"./img/1.bmp",r"./img/2.bmp",r"./img/3.bmp",r"./img/4.bmp",r"./img/5.bmp",r"./img/6.bmp",r"./img/7.bmp",r"./img/8.bmp"]

# Hue
HUE_MIN = 70
HUE_MAX = 170
HUE_MIN_RAMP = HUE_MIN + (HUE_MIN + HUE_MAX)/8
HUE_MAX_RAMP = HUE_MAX - (HUE_MIN + HUE_MAX)/8
# Saturação
SAT_MIN = 10
SAT_MAX = 50
# Value
VAL_MIN = 10
VAL_MAX = 60
# Threshold para ajuste de fundo e frente
T_LOW = 0.45
T_HIGH = 0.95

#===============================================================================
def hue_calc(x):
    """
    Hue - Cálculo do valor do HUE para a seleção do fundo verde
    As seguintes macros definem os limites da função de cálculo: 

                  \/ HUE_MIN_RAMP  \/ HUE_MAX_RAMP
                  __________________     
                 /                  \
        ________/                    \____________
               /\HUE_MIN             /\ HUE_MAX
    
    A função é feita encadeando a função np.where da esquerda para a direita
    em relação a função.

    """
    return np.where(x < HUE_MIN,
                    0, 
                    np.where(x < HUE_MIN_RAMP,
                            ((((HUE_MIN+HUE_MAX)/2)-2*np.absolute(((HUE_MIN+HUE_MAX)/2)-x)))/((HUE_MIN+HUE_MAX)/2), 
                            np.where(x<HUE_MAX_RAMP,
                                     1,
                                     np.where(x < HUE_MAX,
                                              ((((HUE_MIN+HUE_MAX)/2)-2*np.absolute(((HUE_MIN+HUE_MAX)/2)-x)))/((HUE_MIN+HUE_MAX)/2),
                                              0))))

def sat_calc(x):
    """
    Saturação - Cálculo do valor de saturação para a seleção do fundo verde
    As seguintes macros definem os limites da função de cálculo: 
                 \/ SAT_MAX
                  _________     
                 /        
        ________/           
               /\ SAT_MIN    
    """
    x*=100
    return np.where((x > SAT_MIN) & (x < SAT_MAX),
                    (1/(SAT_MAX-SAT_MIN))*x -(SAT_MIN/(SAT_MAX-SAT_MIN)) ,
                    np.where(x < SAT_MIN, 0, 1))

def val_calc(x):
    """
    Value -  Cálculo do valor de value para a seleção do fundo verde
    As seguintes macros definem os limites da função de cálculo: 
                 \/ VAL_MAX
                  _________     
                 /        
        ________/           
               /\ VAL_MIN          
    """
    x*=100
    return np.where((x > VAL_MIN) & (x < VAL_MAX),
                    (1/(VAL_MAX-VAL_MIN))*x -(VAL_MIN/(VAL_MAX-VAL_MIN)), 
                    np.where(x < VAL_MIN, 0 , 1))

def chromakey(img,bg):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Redimensiona o fundo para se adequar a imagem
    bg = cv2.resize(bg, (int(img.shape[1]), int(img.shape[0])))
    alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    alpha = alpha.astype (np.float32)/255

    # Calculando valores
    h = hue_calc(imgHSV[:,:,0])
    s = sat_calc(imgHSV[:,:,1])
    v = val_calc(imgHSV[:,:,2])   
  
    # Inverso do filtro
    alpha = (1.0) - (h*s*v)

    # Threshold fundo
    alpha = np.where(alpha < T_LOW, 0 , alpha)
    # Threshold frente
    alpha = np.where(alpha > T_HIGH, 1 , alpha)

    # Separa o Fundo
    fundo = np.where(alpha < T_LOW, 1 , 0)
    # Separa a frente
    frente = np.where(alpha > T_HIGH, 1 , 0)
    # Todos os valores que não são frente e fundo
    meio = 1 - (fundo + frente)
 
    # Reduzindo verder nas bordas - Onde não é nem fundo e nem frente
    img[:,:,1] = np.where( meio == 1, (img[:,:,0]+img[:,:,2])/2, img[:,:,1])

    # Troca do fundo verde pela imagem de background
    # Cálculo -> Res = Filtro*Imagem + Background*(1-filtro)
    res = np.zeros(img.shape)
    res[:,:,0] = alpha[:,:]*img[:,:,0] + bg[:,:,0]*(1-alpha[:,:])
    res[:,:,1] = alpha[:,:]*img[:,:,1] + bg[:,:,1]*(1-alpha[:,:])
    res[:,:,2] = alpha[:,:]*img[:,:,2] + bg[:,:,2]*(1-alpha[:,:])
        
    # DEBUG
    # cv2.imshow("filtro",alpha)
    # cv2.imshow("resultado",res)
    # cv2.waitKey()

    return res

def main():

    for img in INPUT_IMAGE:
        bg = cv2.imread(r"./img/bg.bmp")
        if bg is None:
            print ('Erro abrindo a imagem de fundo.\n')
            sys.exit ()
        imagem = cv2.imread(img)
        if imagem is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        bg = bg.astype (np.float32) / 255
        imagem = imagem.astype (np.float32) / 255

        print("imagem:", img)
        start_time = timeit.default_timer ()
        # Chamada da função principal
        chroma = chromakey(imagem, bg)
        print ('\tTempo: %f' % (timeit.default_timer () - start_time))
        
        cv2.imwrite (img+'_out.png', chroma*255)

    cv2.destroyAllWindows ()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
