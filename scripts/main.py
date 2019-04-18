import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DIR_PATH = '../data/L16001-L17000/'


# def detecter(img, dp, param1, param2, minDist):
def detecter(img):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=90,
                               param1=100, param2=6, minRadius=13, maxRadius=20)
    # outer_circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
    #                                 param1=140, param2=10, minRadius=440, maxRadius=460)
    outer_circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
                                    param1=140, param2=10, minRadius=440, maxRadius=460)

    circles = np.around(circles[0])
    for i in circles:
        # cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    x = None
    theta = None
    if outer_circle:
        center = np.around(outer_circle[0][0])
        cv2.circle(cimg, (center[0], center[1]), center[2], (0, 255, 0), 2)
        cv2.circle(cimg, (center[0], center[1]), 10, (255, 0, 0), -1)

        # for center in np.around(outer_circle[0]):
        #     cv2.circle(cimg, (center[0], center[1]), center[2], (0, 255, 0), 2)
        #     cv2.circle(cimg, (center[0], center[1]), 10, (255, 0, 0), -1)

        vector = center - circles
        x, theta = cv2.cartToPolar(vector[:, 0], vector[:, 1])
        theta = np.pi - theta

    return x, theta, cimg


def main():
    # Plot pixel histgram -----------------------------------------------
    img = cv2.imread(os.path.join(DIR_PATH, 'L16500.tif'),
                     cv2.IMREAD_GRAYSCALE)
    ax1 = plt.subplot(221)
    ax1.imshow(img)
    ax1.set_title('original')

    # img_mask = cv2.medianBlur(img, 101)
    ret, img_mask = cv2.threshold(img, 100, 100, cv2.THRESH_TOZERO)
    img_mask = cv2.medianBlur(img_mask, 9)
    # ax2 = plt.subplot(222)
    # ax2.imshow(img_mask)
    # ax2.set_title('median blur')

    # plot histgram ----------------------------------------------------
    # array = np.array(img_mask).reshape(-1)
    # freq = {}

    # for i in np.unique(array):
    #     freq[i] = np.count_nonzero(array == i)

    # ax3 = plt.subplot(223)
    # ax3.bar(freq.keys(), freq.values())
    # ax3.set_title('histgram')
    # ax3.set_xlabel('thickness')
    # ax3.set_ylabel('frequency')
    # ------------------------------------------------------------------

    ax2 = plt.subplot(222)
    # THRESH = 100
    # ret, img_mask = cv2.threshold(img_mask, THRESH, THRESH + 2, cv2.THRESH_TOZERO)
    img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0)
    ax2.imshow(img_mask)
    ax2.set_title('adaptiveThreshold')

    img_mask = cv2.medianBlur(img_mask, 15)
    ax3 = plt.subplot(223)
    ax3.imshow(img_mask)
    ax3.set_title('median blur')

    x, theta, cimg = detecter(img_mask)
    # try:
    #     x, theta, cimg = detecter(img_mask)
    # except:
    #     pass

    # E = 51.1
    # r = 617.11
    # d = np.sqrt(150.4 / E) * np.sqrt(r ** 2 + x ** 2) / x

    ax4 = plt.subplot(224)
    ax4.imshow(cimg)
    ax4.set_title('detect circles')
    plt.show()

    # Detect circles with only Canny filter. ----------------------------
    # THRESH = 50

    # for _ in range(5):
    #     THRESH += 2
    #     img = cv2.imread(os.path.join(DIR_PATH, 'L16501.tif'), cv2.IMREAD_GRAYSCALE)
    #     img_mask = cv2.Canny(img, THRESH // 2, THRESH)

    #     plt.figure(figsize=(20, 10))
    #     ax1 = plt.subplot(121)
    #     ax1.imshow(img)

    #     ax2 = plt.subplot(122)
    #     ax2.imshow(img_mask)

    #     plt.savefig(os.path.join('../output/canny', 'canny_{}-{}.png'.format(THRESH // 2, THRESH)))

    # Detect circles ----------------------------------------------------
    # img = cv2.imread(os.path.join(DIR_PATH, 'L16501.tif'), cv2.IMREAD_GRAYSCALE)

    # try:
    #     x, theta, cimg = detecter(img_mask, dp=3, param1=105, param2=6, minDist=80)
    #     # x, theta, cimg = detecter(img, dp=2, param1=100, param2=5, minDist=90)
    # except:
    #     pass

    # E = 51.1
    # r = 617.11
    # d = np.sqrt(150.4 / E) * np.sqrt(r ** 2 + x ** 2) / x

    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot(121, polar=True)
    # ax1.scatter([theta], [d], s=10)

    # # img_mask = cv2.Canny(cimg, 60, 120)
    # # ax1 = plt.subplot(122)
    # # ax1.imshow(img_mask)

    # ax2 = plt.subplot(122)
    # ax2.imshow(cimg)
    # plt.show()

    # Parameter search ---------------------------------
    # for dp in [1, 2, 3]:
    #     for param1 in range(95, 125, 5):
    #         for param2 in range(3, 9):
    #             for minDist in range(50, 100, 10):
    #                 img = cv2.imread(os.path.join(DIR_PATH, 'L16501.tif'), cv2.IMREAD_GRAYSCALE)

    #                 try:
    #                     x, theta, cimg = detecter(img, dp, param1, param2, minDist)
    #                 except:
    #                     continue

    #                 E = 51.1
    #                 r = 617.11
    #                 d = np.sqrt(150.4 / E) * np.sqrt(r ** 2 + x ** 2) / x

    #                 # print('x:', x)
    #                 # print('theta:', math.degrees(theta))
    #                 # print('d:', d)

    #                 plt.figure(figsize=(20, 10))
    #                 ax1 = plt.subplot(121, polar=True)
    #                 ax1.scatter([theta], [d], s=10)

    #                 ax2 = plt.subplot(122)
    #                 ax2.imshow(cimg)
    #                 # plt.show()

    #                 plt.savefig(os.path.join('../output/circle',
    #                                          'circle_dp-{}_param1-{}_param2-{}_minDist-{}.png'.format(
    #                                              dp, param1, param2, minDist)))

    # ----------------------------------------------------

    # cv2.imshow('detected circles', cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ret, img_mask = cv2.threshold(img, THRESH, 255, 0)

    # # Setup SimpleBlobDetector parameters.
    # params = cv2.SimpleBlobDetector_Params()
    # params.minThreshold = THRESH
    # params.maxThreshold = 255

    # detector = cv2.SimpleBlobDetector_create()
    # kp = detector.detect(img_mask)
    # print(kp)
    # # img_mask = cv2.drawMarker(img, keypoints, color=(0, 255, 0))

    # for marker in kp:
    #     img_mask = cv2.drawMarker(img_mask, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
    # plt.imshow(img_mask)
    # plt.show()

    # img_mask = cv2.Canny(img, THRESH, 255)

    # ret, img_mask = cv2.threshold(img, THRESH, 255, 0)
    # img_mask = cv2.medianBlur(img_mask, BLUR)

    # contours = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # img_mask = cv2.drawContours(img_mask, contours, -1, (0, 255, 0), 3)

    # for i, cnt in enumerate(contours):
    #     cnt = np.squeeze(cnt, axis=1)

    #     area = cv2.contourArea(cnt)
    #     if area > 3000:
    #         for c in cnt:
    #             img_mask[c[1], c[0]] = 0

    # fig, ax = plt.subplots(figsize=(10, 10))
    # img_mask = draw_contours(ax, img_mask, contours)
    # plt.show()

    # print(len(contours))

    # imarray = np.array(img_mask)
    # imarray = imarray / 255

    # imgEdge, contours = cv2.findContours(img_mask, 1, 2)

    # for i, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     print(area)

    # cnt = contours[0]
    # area = cv2.contourArea(cnt)
    # print(area)

    # print('shape:', imarray.shape)
    # print('min:', np.min(imarray))
    # print('max:', np.max(imarray))

    # height, width = imarray.shape
    # count = 0

    # for h in range(0, height, 10):
    #     for w in range(0, width, 10):
    #         if imarray[h:h + 3, w:w + 3].all():
    #             if not imarray[h: h + 10, w: w + 10].all():
    #                 imarray[h: h + 10, w: w + 10] = 0
    #         if (40 < h and h < 940) and (250 < w and w < 1150):
    #             imarray[h][w] = 255
    #             if (imarray[h:h + 3, w:w + 3] > 160).all():
    #                 count += 1

    # print(count)
    # print(height * width - count)

    # cv2.namedWindow('imarray', cv2.WINDOW_NORMAL)
    # cv2.imshow('imarray', imarray)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.namedWindow('img_mask', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('img_mask', img_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
