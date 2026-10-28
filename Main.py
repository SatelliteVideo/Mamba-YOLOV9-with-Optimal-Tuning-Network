import os
import cv2 as cv
from Model_FM_YOLOv9_OTN import Model_FM_YOLOv9_OTN
from Model_YoloV3 import Model_YoloV3
from Model_YoloV5 import Model_YoloV5
from Model_YoloV9 import Model_YoloV9
from Motion_Tracking import Motion_Tracking
from Plot_Track import *

# Read Dataset
an = 0
if an == 1:
    path = './Dataset/'
    Out_dir = os.listdir(path)
    Images = []
    tar = []
    for i in range(len(Out_dir)):
        file_path = path + Out_dir[i]
        in_dir = os.listdir(file_path)
        for j in range(len(in_dir)):
            Folder_path = file_path + '/' + in_dir[j]
            folder_dir = os.listdir(Folder_path)
            for k in range(len(folder_dir)):
                print(i, j, k)
                Image_path = Folder_path + '/' + folder_dir[k]
                if folder_dir[k] == 'img':
                    img_dir = os.listdir(Image_path)
                    for s in range(len(img_dir)):
                        print(i, 'Img', s)
                        Img_path = Image_path + '/' + img_dir[s]
                        Image = cv.imread(Img_path)
                        Img = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
                        img_re = cv.resize(Img, [128, 128])
                        tar.append(Out_dir[i])
                        Images.append(img_re)
    Target = np.asarray(tar)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq))).astype('int')
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    np.save('Satelite_Images_all.npy', Images)
    np.save('Target.npy', target)

# Object Detection and Tracking
an = 0
if an == 1:
    bests = np.load('Bestsol.npy', allow_pickle=True)
    Images = np.load('Satelite_Images_all.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    EVAL = []
    EV_Track = []
    Epochs = [5, 10, 15, 20, 25]
    learn = int(Images.shape[0] * 0.75)
    for i in range(len(Epochs)):
        Eval = np.zeros((4, 14))
        Eval_Track = np.zeros((4, 14))
        Eval[0, :], Eval_Track[0, :] = Model_YoloV3(Images, Target)
        Eval[1, :], Eval_Track[1, :] = Model_YoloV5(Images, Target)
        Eval[2, :], Eval_Track[2, :] = Model_YoloV9(Images, Target)
        Eval[3, :], Eval_Track[3, :] = Model_FM_YOLOv9_OTN(Images, Target)
        EVAL.append(Eval)
        EV_Track.append(Eval_Track)
    np.save('Eval_Seg_epoch.npy', EVAL)
    np.save('Eval_Track.npy', EV_Track)


Motion_Tracking()
Plot_Results_Epoch()
Plot_Results_Act()
Plot_Results()
Plot_ROC()