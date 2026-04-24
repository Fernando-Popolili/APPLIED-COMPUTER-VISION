import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# =========================================================
# PIPELINE COMPLETO DE SEGMENTAÇÃO DE REGIÕES CLARAS
# img2, img3, img4, img5
# =========================================================
 
# Dicionário com as imagens
BASE = "C:/Users/labsfiap/Desktop/aula_vision/APPLIED-COMPUTER-VISION/Applied computer vision/Projeto - CP2/"

imagens = {
    "img2": BASE + "img2.png",  # TC cérebro
    "img3": BASE + "img3.png",  # RM cérebro
    "img4": BASE + "img4.png",  # Flores (cinza)
    "img5": BASE + "img5.png"   # Gato (colorida)
}
 
resultados = {}
 
for nome, arquivo in imagens.items():
 
    # -----------------------------------------------------
    # Leitura da imagem
    # -----------------------------------------------------
    img = cv2.imread(arquivo)
    if img is None:
        raise FileNotFoundError(f"Erro ao carregar {arquivo}")
 
    # -----------------------------------------------------
    # Conversão para tons de cinza (quando aplicável)
    # -----------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # -----------------------------------------------------
    # PIPELINE ESPECÍFICO POR IMAGEM
    # -----------------------------------------------------
    if nome == "img2":
        # Equalização de histograma
        eq = cv2.equalizeHist(gray)
        # Threshold para regiões muito claras (borda do crânio)
        _, thresh = cv2.threshold(eq, 200, 255, cv2.THRESH_BINARY)
        # Morfologia: abertura + fechamento
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        aberta = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        fechada = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel, iterations=3)
        # Encontrar maior contorno (borda externa)
        contours, _ = cv2.findContours(
            fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros_like(gray)
        if contours:
            maior = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [maior], -1, 255, -1)
 
    elif nome == "img3":
        # Equalização para destacar a lesão
        eq = cv2.equalizeHist(gray)
        # Threshold mais alto (pegar só regiões MUITO claras)
        _, thresh = cv2.threshold(eq, 210, 255, cv2.THRESH_BINARY)
        # Morfologia
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        aberta = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        fechada = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel, iterations=2)
        # ROI: canto inferior direito
        h, w = fechada.shape
        roi_mask = np.zeros_like(fechada)
        roi_mask[h//2:, w//2:] = 255
        filtrada = cv2.bitwise_and(fechada, roi_mask)
        # Encontrar a lesão (maior região nessa área)
        contours, _ = cv2.findContours(
            filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros_like(gray)
        if contours:
            maior = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [maior], -1, 255, -1)
    elif nome == "img4":
        # Flores → apenas threshold + morfologia
 
        _, binary = cv2.threshold(
            gray, 200, 255, cv2.THRESH_BINARY
        )
 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
 
        mask = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel
        )

    #Gato
    elif nome == "img5":
 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
 
        _, binary = cv2.threshold(
            v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
 
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
 
        # Maior região clara central = gato
        mask = np.zeros_like(gray)
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
 
        segmented = cv2.bitwise_and(img, img, mask=mask)
        resultados[nome] = (img, gray, mask, segmented)
 
    # -----------------------------------------------------
    # Isolamento das regiões claras
    # -----------------------------------------------------
    segmented = cv2.bitwise_and(img, img, mask=mask)
 
    resultados[nome] = (img, gray, mask, segmented)
 
# =========================================================
# VISUALIZAÇÃO DOS RESULTADOS
# =========================================================
linhas = len(resultados)
fig, axes = plt.subplots(linhas, 4, figsize=(16, 4 * linhas))
 
for i, (nome, (orig, gray, mask, seg)) in enumerate(resultados.items()):
 
    axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"{nome} - Original")
 
    axes[i, 1].imshow(gray, cmap="gray")
    axes[i, 1].set_title("Tons de cinza")
 
    axes[i, 2].imshow(mask, cmap="gray")
    axes[i, 2].set_title("Máscara segmentada")
 
    axes[i, 3].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
    axes[i, 3].set_title("Regiões claras isoladas")
 
    for j in range(4):
        axes[i, j].axis("off")
 
plt.tight_layout()
plt.show()