#%%
import cv2
from random import sample


def keypoint_matching(img1, img2, k=2, ratio=0.8):

    #transform the images to grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Initialize the sift detector
    sift = cv2.xfeatures2d.SIFT_create()

    #finding the keypoints and descriptors
    org_kp_1 = sift.detect(gray_img1, None)
    org_kp_2 = sift.detect(gray_img2, None)

    org_kp_1, desc_1 = sift.compute(gray_img1,org_kp_1)
    org_kp_2, desc_2 = sift.compute(gray_img2, org_kp_2)

    #using the Brute-Force matching, with k=2 so the ratio test can be applied
    BruteForce = cv2.BFMatcher()
    all_matches = BruteForce.knnMatch(desc_1, desc_2, k=2)

    #apply the ratio test as explaiend in the sift documentation
    #more info on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    matches = []
    kp_1 = []
    kp_2 = []

    for m,n in all_matches:
        if m.distance < ratio*n.distance:
            matches.append([m])
            kp_1.append(org_kp_1[m.queryIdx])
            kp_2.append(org_kp_2[m.trainIdx])

    return matches, kp_1, kp_2, org_kp_1, org_kp_2


#For plotting drawing the keypoint matches of the boat Images
if __name__ == '__main__':

    boat1 = cv2.imread('images/boat1.pgm')
    boat2 = cv2.imread('images/boat2.pgm')
    
    matches, k1, k2, keypoints_1, keypoints_2 = keypoint_matching(boat1, boat2, k=2, ratio=0.8)
    points = sample(range(0, len(matches)), 10)
    plot_m = []
    for i in points:
        plot_m.append(matches[i])

    drawn_matches_boat = cv2.drawMatchesKnn(boat1,keypoints_1,boat2,keypoints_2,plot_m,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('images/keypoint_matches_boat.jpg', drawn_matches_boat)


