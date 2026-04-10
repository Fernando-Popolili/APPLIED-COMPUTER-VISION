
import cv2
import numpy

PATH = "fachada01.jpg"

im1 = cv2.imread(PATH,1)

img2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(
    nfeatures=150,
    nOctaveLayers=3,
    contrastThreshold=0.05,
    sigma=1.5
)

keypoints, descriptors = sift.detectAndCompute(img2, None)

for kp in keypoints:
    x = int(kp.pt[0])
    y = int(kp.pt[1])
    
    # desenha circulo preenchido vermelho (BGR = (0,0,255))
    cv2.circle(im1, (x, y), radius=2, color=(0, 0, 255), thickness=1)

cv2.namedWindow("Original",2)
cv2.imshow("Original",im1)

cv2.namedWindow("Pontos Principais",2)
cv2.imshow("Pontos Principais",im1)

cv2.waitKey(0)
cv2.destroyAllWindows()