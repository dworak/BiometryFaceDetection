import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# https://www.youtube.com/watch?v=vkWdzWeRfC4
# Jak znajdujemy naroznik w obrazach?
# Jesli suma gradientu dla danego pixela w wertykalnym jak i horyzontalnym kierunku dla danego prostokatnego okna jest
# zdecydowanie duza w porowaniu do jego otoczenia, chodzi tu tak na prawde o lokalne max funkcji gradientu dla danego pixela
# Co jesli polozenie danego pixela, nie pozwala nam dokladnie okreslic, w jakim kierunku suma gradientu jest najwieksza?
# Np. mamy obraz obrocony, szukamy eigen values, to nic innego jak pierwiastki rownania det|A-(lambda-I)|= 0
# gdzie A to macierz odzwierciedlajaca obraz pixeli dla danego okna postaci n x n
# lambda to szukane pierwiastki rownania, tj. eigen values
# I to postac macierzy jednostkowej, takiego wymiaru jak macierz A

def getImageFromCameraCornerHarris():
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        frame[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def getImageFromLocalFileCornerHarris(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
# SLAM (Simultaneous Localization and Mapping)
# Features from Accelerated Segment Test)
# Wybierz piksel ip w obrazie, ktory ma byc zidentyfikowany jako naroznik, czy tez nie.
# Niech jego intensywnosc bedzie Ip
# Wybierz odpowiednia wartosc progowa t
# Teraz, naroznik istnieje jesli istnieje zbior n ciaglych pixeli w okregu 16 pixeli,
# ktore to wszystki sa albo jasniejsze od Ip + t, albo ciemniejsze Ip - t
# Jesli p jest naroznikiem, wowczas conajmniej 3 inne pixele z bliskiego jego otoczenia
# musza byc albo jasniejsze albo ciemniejsze

def getImageFromLocalFileFAST(filepath):
    img = cv2.imread(filepath,0)
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
    print "Threshold: ", fast.getInt('threshold')
    print "NonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    print "Total Keypoints with nonmaxSuppression: ", len(kp)
    fast.setBool('nonmaxSuppression',0)
    cv2.imshow('dst',img2)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    fast.setBool('nonmaxSuppression',0)
    kp = fast.detect(img,None)

    print "Total Keypoints without nonmaxSuppression: ", len(kp)
    img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
    cv2.imshow('dst',img3)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def getImageFromCameraFAST():

    while(True):
        ret, frame = cap.read()
        fast = cv2.FastFeatureDetector()
        fast.setBool('nonmaxSuppression',1)
        kp = fast.detect(frame,None)
        img2 = cv2.drawKeypoints(frame, kp, color=(255,0,0))

        print "Threshold: ", fast.getInt('threshold')
        print "NonmaxSuppression: ", fast.getBool('nonmaxSuppression')
        print "Total Keypoints with nonmaxSuppression: ", len(kp)
        cv2.imshow('frame',img2)
        if cv2.waitKey(1) & 0xff == 27:
            break

    while(True):
        ret, frame = cap.read()
        fast = cv2.FastFeatureDetector()
        fast.setBool('nonmaxSuppression',0)
        kp = fast.detect(frame,None)

        print "Total Keypoints without nonmaxSuppression: ", len(kp)
        img2 = cv2.drawKeypoints(frame, kp, color=(255,0,0))

        print "Threshold: ", fast.getInt('threshold')
        print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
        print "Total Keypoints with nonmaxSuppression: ", len(kp)
        cv2.imshow('frame',img2)
        if cv2.waitKey(1) & 0xff == 27:
            break

#getImageFromLocalFileCornerHarris('./python2/photo.jpg')
#getImageFromCameraCornerHarris()
#getImageFromLocalFileFAST('./python2/photo.jpg')
#getImageFromCameraFAST()