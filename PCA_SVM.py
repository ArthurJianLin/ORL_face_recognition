# 开发人员: ArthurLin
# 开发时间：10:37
# 文件名称：PCA_SVM.py
# 开发工具：PyCharm

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import cv2   #openCV 模块 用于图像处理
import numpy as np
from sklearn.model_selection import train_test_split  #用于切分训练集和测试集
from sklearn.decomposition import PCA  #PCA降维
from sklearn.svm import SVC      #支持向量机
import matplotlib.pyplot as plt
import random


#GUI界面
frameT = Tk()
frameT.geometry('500x200+400+200') #窗口大小
frameT.title('人脸识别') #设置标题
image = Image.open(r'C:\Users\13199\Pictures\2.jpg')
background_image = ImageTk.PhotoImage(image)
w = background_image.width()
h = background_image.height()
background_label = Label(frameT, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

frame1 = Frame(frameT)   #矩形控件 (选择图片+文字框)
frame1.pack(padx=10, pady=10)  # 设置外边距
frame2 = Frame(frameT)   #准确率
frame2.pack(padx=10, pady=10)  # 设置外边距
frame3 = Frame(frameT)   #运行退出按钮
frame3.pack(padx=10, pady=10)
v1 = StringVar()
v2 = StringVar()
ent = Entry(frame1, width=50, textvariable=v1).pack(fill=X, side=RIGHT)  #   文件路径框  x方向填充,靠左
ent = Entry(frame2, width=50, textvariable=v2).pack(fill=X, side=RIGHT) #  文件路径框  x方向填充,靠左
Label(frame2, text="准确率", font=("微软雅黑", 14)).pack(side=LEFT)

def fileopen():
    file_sql = askopenfilename()
    if file_sql:
        v1.set(file_sql)

def match():
    data = []  # 存放图像数据
    label = []  # 存放标签

    # 将40*10的像素为112*92像素的图像处理为400*10302的数组
    for i in range(1, 41):
        for j in range(1, 11):
            path = 'C:/Users/13199/Desktop/face/ORL_Faces/' + 's' + str(i) + '/' + str(j) + '.pgm'
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 灰度读取
            h, w = img.shape
            img_col = img.reshape(h * w)
            data.append(img_col)
            label.append(i)

    C_data = np.array(data)
    C_label = np.array(label)
    # print(C_data.shape)
    # print(C_label)

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(C_data, C_label, test_size=0.9, random_state=256)
    pca = PCA(n_components=10, svd_solver='auto').fit(x_train)

    # PCA降至15维
    C_data_pca = pca.transform(C_data)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Support Vector Classification
    svc = SVC(kernel='linear')
    svc.fit(x_train_pca, y_train)

    # 测试识别准确度
    score = svc.score(x_test_pca, y_test)
    print('%.5f' % score)
    v2.set(score)

    # GUI界面选择图片进行识别
    fig = plt.figure()
    a = fig.add_subplot(121)
    path = v1.get()
    img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    a.imshow(img1, cmap='gray')
    img1 = img1.reshape(h * w)
    img1_pca = pca.transform([img1])    ##一维转二维小技巧
    label_random = svc.predict(img1_pca)
    print(label_random[0])

    a = fig.add_subplot(122)
    path = 'C:/Users/13199/Desktop/face/ORL_Faces/' + 's' + str(label_random[0]) + '/' + str(1) + '.pgm'  # 绘制所选人的第一张图片
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    a.imshow(img2, cmap='gray')
    plt.show()

#按钮事件
btn = Button(frame1, width=7, text='选择图片', font=("微软雅黑", 14), command=fileopen).pack(fill=X, padx=0)
# btn_1 = Button(frame2, width=20, text='匹配文件', font=("宋体", 14), command=fileopen_1).pack(fil=X, padx=10)
ext = Button(frame3, width=10, text='运行', font=("微软雅黑", 14), command=match).pack(fill=X, side=LEFT)
etb = Button(frame3, width=10, text='退出', font=("微软雅黑", 14), command=frameT.quit).pack(fill=Y, padx=0)
frameT.mainloop()