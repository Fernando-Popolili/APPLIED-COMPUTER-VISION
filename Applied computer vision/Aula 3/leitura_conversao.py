import cv2
'''
Importa a biblioteca OpenCV.

cv2 é o módulo principal para:

Leitura de imagens

Conversões de cor

Filtros

Detecção

Processamento

👉 Internamente, OpenCV trabalha com imagens como arrays NumPy.
'''
import matplotlib.pyplot as plt
'''
Importa o módulo de visualização do Matplotlib.

plt é um alias convencional.

Usamos para:

Mostrar imagem

Plotar histograma

Criar gráficos
'''


'''
Define uma função chamada load_image.

path: str é uma anotação de tipo (type hint).

Indica que path deve ser uma string.

Não obriga, mas ajuda em IDEs e organização.
'''
def load_image(path: str):
    image = cv2.imread(path)
    '''
    cv2.imread() lê a imagem do disco.

    Retorna: Um array NumPy 3D → (altura, largura, canais) Ou None se falhar

    Formato retornado: (height, width, 3)

    Importante:
        OpenCV lê imagens no formato BGR, não RGB. Mas o OpenCV usa BGR por padrão.

        Isso é uma decisão histórica da biblioteca.
    '''



    if image is None:
        raise ValueError("Imagem não encontrada.")
    return image # Retorna a imagem carregada.Tipo retornado:numpy.ndarray

'''
Se a imagem não for carregada:

    cv2.imread() retorna None
    Lançamos uma exceção
    Isso evita erro silencioso

Engenharia boa prática:
    Falhar cedo e claramente.
'''

def convert_to_gray(image): # Recebe uma imagem colorida (3 canais).
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #cv2.cvtColor() converte espaço de cor.

'''
COLOR_BGR2GRAY significa:

    Converter de BGR → Escala de Cinza

    O que acontece matematicamente?

Cada pixel:

    Gray = 0.299R + 0.587G + 0.114B

Resultado:

    Imagem 2D

Shape:

    (height, width)


Menos memória.
Mais rápido.
Muito usado em:

    Detecção de borda

    Segmentação

Feature extraction
'''

def show_image(image, title="Imagem"): # Recebe:image title opcional (default = "Imagem")
    plt.imshow(image, cmap='gray') # imshow() mostra a imagem. cmap='gray' indica que é escala de cinza. Se fosse RGB, precisaríamos converter BGR → RGB antes.
    plt.title(title) # Define o título do gráfico.
    plt.axis("off") # Remove eixos (fica mais limpo).
    plt.show() # Renderiza e exibe a imagem. Sem isso, nada aparece.


def plot_histogram(image, title="Histograma"): # Recebe imagem em escala de cinza.
    plt.hist(image.ravel(), bins=256, range=[0,256])
    '''
    image.ravel() - Transforma a matriz 2D em vetor 1D.

        Exemplo:

        Imagem:

        [[0, 10],
        [50, 200]]


        Vira:

        [0, 10, 50, 200]


        Por quê?
        Porque o histograma precisa de uma lista de valores.

            bins=256

        Divide em 256 intervalos.

        Porque imagens 8-bit vão de:

        0 a 255

            range=[0,256]

        Define intervalo de intensidade.
    '''
    plt.title("Histograma de Intensidade") # Título do gráfico.
    plt.xlabel("Intensidade") # Eixo X = valor do pixel
    plt.ylabel("Quantidade de Pixels") # Eixo Y = frequência
    plt.show() # Mostra o gráfico.


'''
Isso é MUITO importante.

Significa:

"Só execute o código abaixo se este arquivo for executado diretamente."

Se ele for importado como módulo, esse bloco não roda.

Isso é padrão profissional.
'''
# ==============================
# Ajuste de brilho e contraste
# ==============================

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    alpha -> controla contraste
    beta  -> controla brilho

    Fórmula aplicada:
    nova_imagem = alpha * imagem + beta
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ==============================
# Equalização de histograma
# ==============================

def equalize_image(image):
    """
    Melhora contraste global redistribuindo intensidades
    """
    return cv2.equalizeHist(image)


# ==============================
# Métricas simples
# ==============================

def calculate_metrics(image):
    brightness = image.mean()
    contrast = image.std()

    return brightness, contrast

# ==========================================
# THRESHOLDING
# ==========================================
'''
Essa é a etapa de decisão: o que é objeto e o que é fundo.

apply_threshold: É um corte seco. Se o pixel for mais brilhante que seu valor escolhido, ele vive (fica branco); se não, ele morre (fica preto).

apply_otsu: É a versão inteligente. O algoritmo de Otsu analisa o histograma da imagem e encontra o ponto exato onde a variância entre as cores do objeto e do fundo é maximizada. É excelente para quando a iluminação da imagem muda.
'''
def apply_threshold(image, threshold=127):
    # Converte imagem contínua em binária usando limiar fixo
    # Pixels acima do limiar viram 255 (branco), abaixo viram 0 (preto)
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def apply_otsu(image):
    # Otsu calcula automaticamente o melhor limiar
    # Maximiza a separação estatística entre duas classes
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


# ==========================================
# MORFOLOGIA
# ==========================================
'''
a função usa o fechamento (MORPH_CLOSE) com um kernel $5 \times 5$.O que faz: Ela primeiro dilata a imagem (expande o branco) e depois a erode (encolhe o branco).Resultado: Isso serve para fechar buracos dentro dos objetos e conectar partes que deveriam estar juntas, mas foram separadas por ruído na binarização.
'''
def apply_morphology(binary_image):
    # Kernel define o tamanho da vizinhança analisada
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # MORPH_CLOSE fecha pequenos buracos e conecta regiões próximas
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return cleaned


# ==========================================
# CONTORNOS E DETECÇÃO
# ==========================================
'''
transforma pixels em dados geométricos.RETR_EXTERNAL: Isso é importante! Indica que você só quer os contornos de fora. Se houver um objeto dentro de outro (como o furo de uma arruela), ele vai ignorar o furo e pegar apenas a borda externa.CHAIN_APPROX_SIMPLE: Economiza memória. Em vez de guardar todos os pontos de uma linha reta, ele guarda apenas os dois pontos das extremidades.boundingRect: Converte a curva complexa do contorno em um retângulo simples $(x, y, largura, altura)$. É o primeiro passo para algoritmos de rastreamento ou reconhecimento de padrões.
'''
def find_contours(binary_image):
    # Detecta bordas das regiões brancas na imagem binária
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_contours(image, contours):   
    image_copy = image.copy() # Evita alterar imagem original

     # Desenha todos os contornos encontrados
    cv2.drawContours(image_copy, contours, -1, (0,255,0), 2)
    return image_copy


def draw_bounding_boxes(image, contours):
    image_copy = image.copy()
    
    for contour in contours:
        # boundingRect cria o menor retângulo que envolve o objeto
        x, y, w, h = cv2.boundingRect(contour)
        # Desenha retângulo ao redor do objeto detectado
        cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0,255,0), 2)

    return image_copy



    


'''
Estrutura de Dados Real

Se você imprimir:

    print(type(img))
    print(img.shape)
    print(img.dtype)


        Você verá algo como:

        <class 'numpy.ndarray'>
        (720, 1280, 3)
        uint8


        Significado:

        720 linhas

        1280 colunas

        3 canais

        Cada pixel ocupa 8 bits (0–255)

Como explicar isso em sala (nível alto)

Você pode dizer:

        "Uma imagem nada mais é do que um tensor de ordem 3."

        Colorida:

        H × W × 3


        Cinza:

        H × W


        E visão computacional começa entendendo isso profundamente.
'''
