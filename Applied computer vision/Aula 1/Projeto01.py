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
    img = carregarImagem('Aula 1/imagem.jpg')
    gray = converterCinza(img)
    mostrarImagem(gray, "Imagem em escala de cinza")
    histograma(gray)