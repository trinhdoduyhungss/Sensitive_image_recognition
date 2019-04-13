import os
import cv2
import numpy as np
path_data = './'
testaccuracy = []
data_train = []
img2 = cv2.imread('cute2jpg.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(img2, (20, 20))
for i_name in os.listdir(os.path.join(path_data, 'testdata/')):
            print(i_name)
            var_img = cv2.imread(path_data+'testdata/'+i_name)
            var_img = cv2.resize(var_img, (20, 20))
            var_img = cv2.cvtColor(var_img,cv2.COLOR_BGR2GRAY)
            data_train.append(var_img)
x = np.array(data_train)###
np.save('x.npy',x)
x2= np.array(img2)
train = x[:,:50].reshape(-1,400).astype(np.float32)
test2 = x2.reshape(-1,400).astype(np.float32)
k = np.arange(3)
train_labels = np.repeat(k, 118)[:, np.newaxis]
knn = cv2.ml.KNearest_create()
knn.train(train, 0, train_labels)
ret, results, neighbours, dist = knn.findNearest(test2, 5)
for nei in neighbours[0]:
    testaccuracy.append(nei)
accuracy = testaccuracy.count(results[0][0]) * 100.0 / len(testaccuracy)
if results == [0.]: print ({'label': "Ảnh đẹp. ", 'accuracy': "Độ chính xác : " + str(accuracy)})
if results == [1.]: print ({'label': "Ảnh xấu. ", 'accuracy': "Độ chính xác : " + str(accuracy)})
if results == [2.]: print ({'label': "Ảnh bình thường. ", 'accuracy': "Độ chính xác : " + str(accuracy)})