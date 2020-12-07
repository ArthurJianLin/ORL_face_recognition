# 开发人员: ArthurLin
# 开发时间：17:06
# 文件名称：ORL_PCA.py
# 开发工具：PyCharm

#导入模块
import cv2   #openCV 模块 用于图像处理
import numpy as np
from sklearn.model_selection import train_test_split  #用于切分训练集和测试集
from sklearn.decomposition import PCA  #PCA降维
from sklearn.svm import SVC      #支持向量机
import matplotlib.pyplot as plt
import random

data=[]#存放图像数据
label=[]#存放标签

#将40*10的像素为112*92像素的图像处理为400*10302的数组
for i in range(1,41):
    for j in range(1,11):
        path='C:/Users/13199/Desktop/face/ORL_Faces/'+'s'+str(i)+'/'+str(j)+'.pgm'
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)      #灰度读取
        h,w=img.shape
        img_col=img.reshape(h*w)
        data.append(img_col)
        label.append(i)

C_data=np.array(data)
C_label=np.array(label)
# print(C_data.shape)
# print(C_label)

#切分数据集
x_train,x_test,y_train,y_test=train_test_split(C_data,C_label,test_size=0.2,random_state=256)
pca=PCA(n_components=15,svd_solver='auto').fit(x_train)

#PCA降至15维
C_data_pca=pca.transform(C_data)
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)

#Support Vector Classification
svc=SVC(kernel='linear')
svc.fit(x_train_pca , y_train)

#测试识别准确度
print('%.5f'%svc.score(x_test_pca,y_test))

#选取随机图片
num = random.randint(0,399)
Per_num = num//10 + 1  #第几个人
Pic_num = num%10 + 1   #第几张图片
# print(num+1)
# print(Per_num)
# print(Pic_num)
##绘制灰度图
fig = plt.figure()
a = fig.add_subplot(121)
path='C:/Users/13199/Desktop/face/ORL_Faces/'+'s'+str(Per_num)+'/'+str(Pic_num)+'.pgm'  #绘制所选择图片
img1=cv2.imread(path)
a.imshow(img1,cmap='gray')

label_random = svc.predict([C_data_pca[num+1,:]])    ##一维转二维小技巧
print(label_random[0])
a = fig.add_subplot(122)
path='C:/Users/13199/Desktop/face/ORL_Faces/'+'s'+str(label_random[0])+'/'+str(1)+'.pgm'  #绘制所选人的第一张图片
img2=cv2.imread(path)
a.imshow(img2,cmap='gray')
plt.show()


# ##选取测试样本进行测试
# # label_test1 = svc.predict(C_data_pca)
# label_test2 = svc.predict([C_data_pca[0,:]])    ##一维转二维小技巧
# # print(label_test2)
