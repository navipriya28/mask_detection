import numpy as np
import cv2
from sklearn.svm import SVC
#supply vector machine
#supply vector classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')

with_mask.shape
#[199,50,50,3]

#to convert 4d into 2d
with_mask=with_mask.reshape(199,50*50*3)
without_mask=without_mask.reshape(199,50*50*3)
#[398,7500]

#concatenating _r rows
X=np.r_[with_mask,without_mask]

#setting 398 to zeros
labels=np.zeros(X.shape[0])

#from 200 to the rest of data is 1.0(i.e,without mask)
labels[200:]=1.0

names={0:'Mask',1:'No Mask'}

#split the data for testing(x,y,testing size)
x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.25)
#x_train.shape(300,7500)

#3D-3 colums only(reduction of dimensionality)
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)

x_train[0]
x_train.shape
#(300,3)

#ml 
svm=SVC()
svm.fit(x_train,y_train)

#convert into 3d
x_test=pca.transform(x_test)

y_pred=svm.predict(x_test)

#(actual_data,pred)
accuracy_score(y_test,y_pred)
#0.98

haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
data=[]
color=(0,255,0) 
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=svm.predict(face)[0]
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            print(n)
            if n=='No Mask':
                color=(0,0,255)
            else:
                color=(0,255,0)
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27:
            break
capture.release()
cv2.destroyAllWindows()
