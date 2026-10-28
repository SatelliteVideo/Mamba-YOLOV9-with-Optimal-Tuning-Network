import numpy as np
import cv2 as cv

no_of_dataset = 3

def Motion_Tracking():
    for a in range(no_of_dataset):
        if a == 0:
            Images = np.load('Motion_Tracked_Car.npy', allow_pickle=True)
        elif a == 1:
            Images = np.load('Motion_Tracked_plane.npy', allow_pickle=True)
        else:
            Images = np.load('Motion_Tracked_ship.npy', allow_pickle=True)
        for i in range(len(Images)):
            Image = Images[i]
            cv.imshow('Motion_Tracking', Image)
            if cv.waitKey(50) & 0xFF == ord('q'):
                break
        cv.waitKey(0)

if __name__ == '__main__':
    Motion_Tracking()