#Coverter imagem colorida numa escala cinza
 
#Biblioteca OpenCV - usada para manipular as imagens
import cv2
 
#Biblioteca para a exibição de elementos graficos
import matplotlib.pyplot as plt
 
 
 
#BLOCO DE FUNÇÕES PARA MANIPULAR A IMAGEM#
'''
1° passo: definir a função
'''
def carregarImagem(path):
    imagem = cv2.imread(path)
    #Lendo a imagem     
    if imagem is None:
        raise ValueError('Imagem não encontrada')
    return imagem

'''
2° Passo: Definir a função que converte a escala para cinza
'''
def converterCinza(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

'''
3° Passo: Definir a função de histograma
'''
def histograma(imagem):
    plt.hist(imagem.ravel(), bins=256, range=[0,256])
    plt.title("Histograma de Intensidade")
    plt.xlabel("Intensidade")
    plt.ylabel("Quantidade de pixels")
    plt.show()
'''
4° Passo: Definir a função que exibe a imagem
'''
def mostrarImagem(imagem, title="Imagem"):
    plt.imshow(imagem, cmap='gray') #ImgShow -> Mostra a imagem;
                                    #cmap -> Indica que é um mapeamento em escala em cinza
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    img = carregarImagem('imagem.jpg')
    gray = converterCinza(img)
    mostrarImagem(gray, "Imagem em escala de cinza")
    histograma(gray)

#==============================================

#Ajuste de brilho e contraste da imagem

#==============================================

def ajustarBrilhoContraste(imagem, alpha=1.0, beta=0):
    '''
    Alpha -> Controla o contraste da imagem
    Beta -> Controla o brilho
    '''
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

#==============================================

#Equalização do histograma

#==============================================

def equalizar_histograma(imagem):
    '''
    Melhorar contraste global redistribuindo a intensidade dos pixels
    '''
    return cv2.equalizeHist(imagem)

#==============================================

#Equalização do histograma

#==============================================

def calcular_metricas(imagem):
    brilho = imagem.mean() 
    contraste = imagem.std()# metrica será observada a partir do desvio padrao

    return brilho, contraste

