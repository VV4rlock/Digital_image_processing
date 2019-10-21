import numpy as np
import matplotlib.pyplot as plt
import cv2

MIN_W, MIN_H = 50, 70
S_bound = MIN_H * MIN_W
PATH = r'/home/warlock/DIP/1/segment.jpeg'

def show_hsv_image(img: np.ndarray): #back from hsv to bgr
    bgr_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('original image', bgr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(img):
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_color_mask_inv(img, color, disp=0.05) -> np.ndarray:
    lower_bound = color * (1 - disp)
    upper_bound = color * (1 + disp)
    print("mask boundaries: {} - {}".format(lower_bound, upper_bound))
    mask = cv2.inRange(img, lower_bound, upper_bound)
    #print(mask)
    return (mask + 1) * 255 # invert


def get_image_color_mask(img, color, disp=0.05) -> np.ndarray:
    lower_bound = color * (1 - disp)
    upper_bound = color * (1 + disp)
    print("mask boundaries: {} - {}".format(lower_bound, upper_bound))
    mask = cv2.inRange(img, lower_bound, upper_bound)
    # print(mask)
    return mask


if __name__ == "__main__":
    img = cv2.imread(PATH)
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    print("shape original: {}, example[0][0]={}".format(img.shape, img[0][0]))

    '''
    print(img.shape)
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    print("shape HSV: {}, example[0][0]={}".format(HSV_img.shape, HSV_img[0][0]))

    #YELLOW_HSV = HSV_img[0][0] # left angle is yellow
    #mask = get_image_color_mask_inv(HSV_img, YELLOW_HSV, disp=np.array([0.15, 0.3, 0.5]))

    show_image(HSV_img)
    GREEN_HSV = [100, 180, 55 ]#HSV_img[392][784]
    # print(GREEN_HSV)
    mask = get_image_color_mask(HSV_img, GREEN_HSV, disp=np.array([0.25, 0.5, 0.7]))
    #mask = cv2.bitwise_and(mask, mask2)


    print("mask shape: {}, exampe: {}".format(mask.shape, mask[0][0]))
    print(mask)
    masked_image = cv2.bitwise_and(HSV_img, HSV_img, mask=mask)
    show_hsv_image(masked_image)

    '''
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    '''
    #print(cnts.shape)
    #print(hierarchy.shape)
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image
    cv2.drawContours(img_contours, cnts, -1, (0, 255, 0), 3)

    cv2.imshow("contours", img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > S_bound:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)  # mark on orig image

    #show_hsv_image(HSV_img)
    cv2.imshow("final", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #res = cv2.findContours(HSV_img, cv2.RETR_CCOMP)
    #print(len(res))
    # displaying the Hsv format image


