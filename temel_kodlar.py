# %%

# =============================================================================
# RESMİ İÇE AKTARMA
# =============================================================================
import cv2

# içe aktar
img = cv2.imread("cat_img1.jpg", 0) 
# görselleştirme
cv2.imshow("Kedi", img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
# %%

# =============================================================================
# VİDEO İÇE AKTARMA
# =============================================================================
import cv2
import time

# video ismi
kayak_video = "20216611-uhd_3840_2160_30fps.mp4"

# içe aktar
cap = cv2.VideoCapture(kayak_video)

# video boyutunu öğrenme
print("Genişlik: ",cap.get(3))
print("Yükseklik: ",cap.get(4))

# video kaynağı başarılı bir şekilde açıldı mı?
if cap.isOpened() == False:
    print("Hata")
    
# sürekli olarak çerçeve okuma
while True:
    ret,frame = cap.read()
    
    if ret == True:
        
        time.sleep(0.01)
        
        cv2.imshow("Kayak Video", frame)
    else:
        break

cap.release() # stop capture
cv2.destroyAllWindows()
# %%

# =============================================================================
# KAMERA AÇMA VE VİDEO KAYDI
# =============================================================================
import cv2

# kamerayı içe aktar
cap = cv2.VideoCapture(0)

# genişlik ve yükseklik bilgisi alma
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # cap.get()
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # cap.get()
print(width,height)

# rec
writer = cv2.VideoWriter("kamera_kaydi.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width, height)) #videoWriter()

while True:
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    
    # save
    writer.write(frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
# %%

# =============================================================================
# YENİDEN BOYUTLANDIR VE KIRP
# =============================================================================
import cv2

img = cv2.imread("cat_img1.jpg")

# boyut öğrenme
print("Kedi",img.shape)
cv2.imshow("Orijinal",img)

# resized (yeniden boyutlandırma)
imgResized = (cv2.resize(img, (800,800)))
cv2.imshow("Img Resized", imgResized)

# kırpma
imgCropped = img[:200, 0:300] # width height --> height width
cv2.imshow("Img Cropped", imgCropped)
# %%

# =============================================================================
# ŞEKİLLER VE METİN
# =============================================================================
import cv2
import numpy as np

# cv2.FILLED şeklin içini doldurur

# resim oluştur
img = np.zeros((512,512,3), np.uint8) # siyah bir resim # zeros(), uint
cv2.imshow("Siyah",img)

# çizgi
# (resim, başlangıç noktası, bitiş noktası, renk, kalınlık)
cv2.line(img, (100,100), (100,300), (0,255,0), 3)
cv2.imshow("Line",img)

# dikdörtgen
# (resim, başlangıç noktası, bitiş noktası, renk)
cv2.rectangle(img, (0,0), (256,256), (255,0,0))
cv2.imshow("Rect",img)

# çember
# (resim,merkez, yarıçap, renk)
cv2.circle(img, (300,300), 45, (0,0,255))
cv2.imshow("Circle",img)

# metin
# (resim, metin, başlangç noktası, font, kalınlık, renk)
cv2.putText(img, "OpenCV", (350,350), cv2.FONT_ITALIC, 1, (255,255,255))
cv2.imshow("Metin",img)
# %%

# =============================================================================
# GÖRÜNTÜLERİN BİRLEŞTİRİLMESİ
# =============================================================================
import cv2
import numpy as np

img = cv2.imread("cat_img1.jpg")
cv2.imshow("Cat",img)

# yatay
hor = np.hstack((img,img)) # hstack()
cv2.imshow("Horizontal",hor)

# dikey
ver = np.vstack((img,img)) # vstack()
cv2.imshow("Vertical",ver)
# %%

# =============================================================================
# PERSPEKTİF ÇARPITMA
# =============================================================================
import cv2
import numpy as np

img = cv2.imread("cat_img1.jpg")
cv2.imshow("Cat",img)

# yeni resmin boyutları
width = 400
height = 500

# pts1 ilk resmi ifade ediyor (köşe koordinatlarını), pts2 ikinci resmi
pts1 = np.float32([[203,1],[1,472],[540,150],[338,617]])
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

# dönüşüm değerlerini hesapla
matrix = cv2.getPerspectiveTransform(pts1, pts2) # cv2.getPerspectiveTransform
print(matrix)

# dönüştürülmüş resim
imgOutPut = cv2.warpPerspective(img, matrix, (width,height)) # cv2.warpPerspective
cv2.imshow("Yeni resim", imgOutPut)
# %%

# =============================================================================
# GÖRÜNTÜLERİ KARIŞTIRMAK
# =============================================================================
import cv2
import matplotlib.pyplot as plt

# görüntüleri yükle ve (matplotlib ve cv farklı renk kodları üzerine çalışır)
# renk dönüşümü yap

img1 = cv2.imread("cat_img1.jpg")
img1 =  cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("dog-7514202_640.jpg")
img2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# yeni figür oluştur
plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# boyut bilgilerini al
print(img1.shape)
print(img2.shape)

# yeniden boyutlandır
img1 = cv2.resize(img1, (600,600))
img2 = cv2.resize(img2, (600,600))

# karıştırılmış resim alpha*img1 + beta*img2
y_resim = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0) # addWeighted()
plt.figure()
plt.imshow(y_resim)
# %%

# =============================================================================
# GÖRÜNTÜ EŞİKLEME
# =============================================================================
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat_img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# eşikleme
_, thresh_img = cv2.threshold(img, thresh = 120, maxval = 255, type = cv2.THRESH_BINARY) # threshold

# uyarlamalı eşik değeri
thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8) # adaptiveThreshold

plt.figure()
plt.imshow(thresh_img2, cmap = 'gray')
plt.axis("off")
plt.show()
# %%

# =============================================================================
# BULANIKLAŞTIRMA
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# blurring (detayı azaltır, gürültüyü engeller)
img = cv2.imread("cat_img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Orijinal görüntü
plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("Original Img"), plt.show()

"""
- Ortalama bulanıklaştırma yöntemi
"""
ort_img = cv2.blur(img, ksize=(3, 3))
plt.figure(), plt.imshow(ort_img), plt.axis("off"), plt.title("Ort Bulanıklaştırma"), plt.show()

"""
- Gaussian blur
"""
gb = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=7)
plt.figure(), plt.imshow(gb), plt.axis("off"), plt.title("Gaussian Blur"), plt.show()

"""
- Medyan blur
"""
mb = cv2.medianBlur(img, ksize=3)
plt.figure(), plt.imshow(mb), plt.axis("off"), plt.title("Medyan Blur"), plt.show()

# Gaussian gürültüsü ekleme
def gaussianNoise(image): 
    w, h, c = image.shape  # resmin boyutları
    mean = 0                # ortalama
    var = 0.05              # varyans
    sigma = var ** 0.5      # standart sapma hesaplanır
    
    gauss = np.random.normal(mean, sigma, (w, h, c))  # rastgele gürültü
    noisy = np.clip(image + gauss, 0, 1)              # Değerleri 0-1 arasında sınırlama
    
    return noisy

# Normalize etme
img_normalized = img / 255.0  # 0-1 aralığına normalize et
plt.figure(), plt.imshow(img_normalized), plt.axis("off"), plt.title("Normalized Original"), plt.show()

gaussianNoisyImage = gaussianNoise(img_normalized)
plt.figure(), plt.imshow(gaussianNoisyImage), plt.axis("off"), plt.title("Gaussian Noisy"), plt.show()

# Gaussian blur
gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize=(3, 3), sigmaX=7)
plt.figure(), plt.imshow(gb2), plt.axis("off"), plt.title("With Gaussian Blur"), plt.show()

# Salt and Pepper noise
def saltPepperNoise(image):
    w, h, c = image.shape
    s_vs_p = 0.5  # tuz ve karabiber gürültüsü oranı
    amount = 0.004  # gürültünün görüntüde ne kadar yer kaplayacağı
    noisy = np.copy(image)  # girdi görüntüsünün bir kopyasını oluşturur
    
    # salt beyaz
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1
    
    # pepper siyah
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy

spImage = saltPepperNoise(img_normalized)
plt.figure(), plt.imshow(spImage), plt.axis("off"), plt.title("Salt and Pepper"), plt.show()

mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize=3)
plt.figure(), plt.imshow(mb2), plt.axis("off"), plt.title("With Median Blur"), plt.show()
# %%

# =============================================================================
# MORFOLOJİK OPERASYONLAR
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat_img1.jpg",0)
plt.figure(), plt.imshow(img, cmap = 'gray'), plt.axis("off"), plt.title("Cat")

# erozyon
kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result, cmap = "gray"), plt.axis("off"), plt.title("Erozyon")  

# genişleme dilation
result2 = cv2.dilate(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result2, cmap = "gray"), plt.axis("off"), plt.title("Genişleme")   

# white noise
whiteNoise = np.random.randint(0,2, size = img.shape[:2])
whiteNoise = whiteNoise*255
plt.figure(), plt.imshow(whiteNoise, cmap = "gray"), plt.axis("off"), plt.title("White Noise")

noise_img = whiteNoise + img
plt.figure(), plt.imshow(noise_img, cmap = "gray"), plt.axis("off"), plt.title("Img w White Noise")

# açılma
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Açılma")

# black noise
blackNoise = np.random.randint(0,2, size = img.shape[:2])
blackNoise = whiteNoise *- 255
plt.figure(), plt.imshow(blackNoise, cmap = "gray"), plt.axis("off"), plt.title("Black Noise")

black_noise_img = blackNoise + img
black_noise_img[black_noise_img<=-245] = 0
plt.figure(), plt.imshow(black_noise_img, cmap = "gray"), plt.axis("off"), plt.title("Black Noise Img")

# kapatma
closing = cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(closing, cmap = 'gray'), plt.axis("off"), plt.title("Kapatma")

# gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(), plt.imshow(gradient, cmap = "gray"), plt.axis("off"), plt.title("Gradyan")
# %%

# =============================================================================
# GRADYANLAR
# =============================================================================
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat_img1.jpg",0)
plt.figure(), plt.imshow(img, cmap = 'gray'), plt.axis("off"), plt.title("Cat")

# x gradyan
sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelx, cmap = "gray"), plt.axis("off"), plt.title("Sobel X")

# y gradyan
sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobely, cmap = "gray"), plt.axis("off"), plt.title("Sobel Y")

# laplacian gradian
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure(), plt.imshow(laplacian, cmap = "gray"), plt.axis("off"), plt.title("Laplacian")
# %%

# =============================================================================
# HİSTOGRAM
# =============================================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("first.png")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img_vis)

print(img.shape)

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
print(img_hist.shape)
plt.figure(), plt.plot(img_hist)

color = ('b','g','r')
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist, color = c)
    
keddy = cv2.imread("cat_img1.jpg")
keddy_vis = cv2.cvtColor(keddy, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(keddy_vis)

print(keddy.shape)

mask = np.zeros(keddy.shape[:2], np.uint8)
plt.figure(), plt.imshow(mask, cmap = "gray")


# %%

# =============================================================================
# KENAR ALGILAMA
# =============================================================================
# %%

# =============================================================================
# KÖŞE ALGILAMA
# =============================================================================
# %%

# =============================================================================
# KONTUR ALGILAMA
# =============================================================================
# %%

# =============================================================================
# RENK İLE NESNE TESPİTİ
# =============================================================================
# %%

# =============================================================================
# ŞABLON EŞLEME
# =============================================================================
# %%

# =============================================================================
# ÖZELLİK EŞLEŞTİRME
# =============================================================================
# %%

# =============================================================================
# HAVZA ALGORİTMASI
# =============================================================================
# %%

# =============================================================================
# YÜZ TANIMA PROJESİ
# =============================================================================
# %%

# =============================================================================
# KEDİ YÜZÜ TANIMA PROJESİ
# =============================================================================
# %%

# =============================================================================
# ÖZEL BENZER ÖZELLİKLER İLE NESNE ALGILAMA
# =============================================================================
# %%

# =============================================================================
# YAYA TESPİTİ
# =============================================================================
# %%

# =============================================================================
# ORTALAMA KAYMA ALGORİTMASI
# =============================================================================
# %%

# =============================================================================
# KEŞİFSEL VERİ ANALİZİ
# =============================================================================
# %%

# =============================================================================
# TAKİP ALGORİTMALARI
# =============================================================================
# %%

# =============================================================================
# ÇOKLU NESNE TAKİBİ
# =============================================================================