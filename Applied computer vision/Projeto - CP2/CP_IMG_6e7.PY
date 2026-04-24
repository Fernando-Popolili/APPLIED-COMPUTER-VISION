import cv2
import numpy as np
import matplotlib.pyplot as plt
 


# DETECÇÃO DE LINHAS E PONTOS DE INTERESSE
# Imagens img6 e img7
# =====================================================

BASE = "C:/Users/labsfiap/Desktop/aula_vision/APPLIED-COMPUTER-VISION/Applied computer vision/Projeto - CP2/"

imagens = {
    "img6": BASE + "img6.png",
    "img7": BASE + "img7.png"
}
 
resultados = {}
 
for nome, arquivo in imagens.items():
 
    # ---------------------------------------------
    # Leitura da imagem
    # ---------------------------------------------
    img = cv2.imread(arquivo)
    if img is None:
        raise FileNotFoundError(f"Erro ao carregar {arquivo}")
 
    img_color = img.copy()
 
    # ---------------------------------------------
    # Pré-processamento
    # ---------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # ---------------------------------------------
    # Segmentação por bordas (Canny)
    # ---------------------------------------------
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
 
    # ---------------------------------------------
    # DETECÇÃO DE LINHAS – HOUGH
    # ---------------------------------------------
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=80,
        maxLineGap=10
    )
 
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(
                img_color,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )
 
    # ---------------------------------------------
    # DETECÇÃO DE PONTOS – HARRIS CORNER
    # ---------------------------------------------
    gray_float = np.float32(gray)
    harris = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )
 
    harris = cv2.dilate(harris, None)
 
    # Marcar os pontos mais fortes
    img_color[harris > 0.01 * harris.max()] = [255, 0, 0]
 
    resultados[nome] = (img, edges, img_color)
 
# =====================================================
# VISUALIZAÇÃO DOS RESULTADOS
# =====================================================
fig, axes = plt.subplots(len(resultados), 3, figsize=(15, 6))
 
for i, (nome, (orig, edges, final)) in enumerate(resultados.items()):
 
    axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"{nome} - Original")
 
    axes[i, 1].imshow(edges, cmap="gray")
    axes[i, 1].set_title("Bordas (Canny)")
 
    axes[i, 2].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    axes[i, 2].set_title("Linhas (vermelho) e Pontos (azul)")
 
    for j in range(3):
        axes[i, j].axis("off")
 
plt.tight_layout()
plt.show()