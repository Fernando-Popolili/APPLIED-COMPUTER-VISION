import cv2
import numpy as np

# 1. Corrigido com 'r' antes das aspas para tratar as barras corretamente
path = r"C:\Users\labsfiap\Desktop\aula\APPLIED-COMPUTER-VISION\Applied computer vision\Aula 4\imagem.jpg"

# Carrega em escala de cinza (o 0 é fundamental para o OTSU)
img1 = cv2.imread(path, 0)


# 2. O threshold devolve DOIS valores: o limite usado e a imagem binária
ret, img_bin = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          
kernel = np.ones((3,3), np.uint8)
    
# Processamentos
img_erode1 = cv2.erode(img1, kernel, iterations=1)
img_erode2 = cv2.erode(img_erode1, kernel, iterations=2)
    
# 3. Usamos a imagem binária (img_bin) para a dilatação, não a tupla do threshold
img_dilate = cv2.dilate(img_bin, kernel, iterations=1)

# Janelas e exibição
cv2.imshow("Original", img1)
cv2.imshow("Binaria (OTSU)", img_bin)
cv2.imshow("Erodida (Cinza)", img_erode1)
cv2.imshow("Dilatada (da Binaria)", img_dilate)

cv2.waitKey(0)
cv2.destroyAllWindows()
