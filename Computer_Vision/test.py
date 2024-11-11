import cv2

img = cv2.imread("input.jpg")
def show(img):
    cv2.imshow("image",img)
    cv2.waitKey(0)
print(img.shape)
img2 = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
cv2.imshow("image",img2)
cv2.waitKey(0)
img4 = cv2.flip(img, 0)
show(img4)
