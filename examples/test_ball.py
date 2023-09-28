import cv2
import numpy as np

# 读入图片
img = cv2.imread("1.png")
# 中值滤波，去噪
img = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)


# # 阈值分割得到二值化图片
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # 膨胀操作
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# bin_clo = cv2.dilate(binary, kernel2, iterations=2)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("final image",gray)
# print(gray.shape)
# gray.dtype = np.uint8
binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
# cv2.imshow("final image",binary)
# binary = cv2.bitwise_not(binary)
# output = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_16U)
# # 连通域分析
# print(bin_clo.shape)
gray = cv2.bitwise_not(gray)
cv2.imshow('original', gray)


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)

# 查看各个返回值
# 连通域数量
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
print('stats = ',stats)
# 连通域的中心点
print('centroids = ',centroids)
# 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
print('labels = ',labels)

# 不同的连通域赋予不同的颜色
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num_labels):

    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
cv2.imshow('oginal1', output)
cv2.waitKey()
cv2.destroyAllWindows()
