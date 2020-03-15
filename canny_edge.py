import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import logging
from disjoin_set import DisjointUnionManager
from math import exp

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
IMAGE_PATH = r"DIP/silv.jpg"
SMALL_SCREEN = False
GAUSS_SIZE = 7
GAUSS_STD = 2
HIGH_THRESHOLD = 100
LOW_THRESHOLD = 40
pi = np.pi

gauss = signal.gaussian(GAUSS_SIZE, GAUSS_STD).reshape(-1, 1)
gauss = gauss / gauss.sum()
#print(gauss)
gauss_filter = np.dot(gauss, gauss.T)
#gauss_filter = gauss_filter / (gauss_filter.sum())
#print(gauss_filter, gauss_filter.sum())


second_derivative = np.array([-1, 2, -1]).reshape(-1, 1)
derivate_filter = np.dot(second_derivative, second_derivative.T)

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])
SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
])

def apply_filter(image, filter) -> np.ndarray:
    return cv2.filter2D(image, -1, filter)

def Canny_edge(grey_img: np.ndarray):
    show_image(grey_image)
    image = apply_filter(grey_image, gauss_filter)
    # image = apply_filter(image, gauss_filter)
    y = apply_filter(image.astype(np.float32), SOBEL_Y)
    x = apply_filter(image.astype(np.float32), SOBEL_X)
    magn = np.sqrt(x ** 2 + y ** 2)
    #show_image(magn)
    one_pix = np.zeros(magn.shape)
    #print(x.min(), x.max(), y.min(),y.max())
    angles = np.arctan2(y, x)
    round_set = {-pi:(0,-1),
                 -pi * 3/4:(1,-1) ,
                 -pi /2:(1,0),
                 -pi/4:(1,1),
                 0:(0,1),
                 pi/4:(-1,1),
                 pi/2:(-1,0),
                 pi * 3/4:(-1,-1),
                 pi:(0,-1)}
    print(round_set)
    pi_8 = pi/8
    for i in range(1, angles.shape[0] - 1):
        for j in range(1, angles.shape[1] - 1):
            direction = None
            cur = angles[i,j]

            for phi in round_set:
                if abs(cur - phi) < pi_8:
                    direction = round_set[phi]
                    break
            if direction is None:
                print("wtf {} {}".format(cur, round_set))
                exit(1)
            if magn[i, j] > magn[i + direction[0], j + direction[1]] and magn[i, j] > magn[i - direction[0], j - direction[1]]:
                one_pix[i, j] = magn[i, j]

            '''
            change = True
            for r in round_set:
                dir = round_set[r]
                if magn[i,j] < magn[i+dir[0],j+dir[1]]:
                    change = False
                    break
            if change:
                one_pix[i, j] = magn[i, j]
            '''
    show_image(one_pix)

    one_pix[one_pix < LOW_THRESHOLD] = 0

    show_image(one_pix)

    disjoint_set = DisjointUnionManager()
    if one_pix[0, 0] > 0:
        one_pix[0, 0] = disjoint_set.create_new_union(one_pix[0, 0])
    for j in range(1, one_pix.shape[1]):
        if one_pix[0, j] > 0:
            if one_pix[0, j - 1] > 0:
                one_pix[0, j] = one_pix[0, j - 1]
                element = disjoint_set.get_union_by_number(one_pix[0, j - 1])
                element.set_value(max(one_pix[0, j - 1], element.get_value()))
            else:
                one_pix[0, j] = disjoint_set.create_new_union(one_pix[0, j])
    for i in range(1, one_pix.shape[0]):
        for j in range(one_pix.shape[1]):
            if one_pix[i, j] > 0:
                marks = [
                    one_pix[i,     j - 1] if j > 0 and one_pix[i    , j - 1] > 0 else 0,
                    one_pix[i - 1, j - 1] if j > 0 and one_pix[i - 1, j - 1] > 0 else 0,
                    one_pix[i - 1, j    ] if one_pix[i - 1, j    ] > 0 else 0,
                    one_pix[i - 1, j + 1] if j + 1 < one_pix.shape[1] and one_pix[i - 1, j + 1] > 0 else 0,
                ]
                cur = 0
                for mark in marks:
                    if mark > 0:
                        if cur > 0:
                            el = disjoint_set.get_union_by_number(cur)
                            el.union(disjoint_set.get_union_by_number(mark))
                        cur = mark
                if cur == 0:
                    one_pix[i, j] = disjoint_set.create_new_union(one_pix[i, j])
                else:
                    element = disjoint_set.get_union_by_number(cur)
                    element.set_value(max(one_pix[i,j], element.get_value()))
                    one_pix[i, j] = cur
    print(one_pix.max())
    for i in range(1, one_pix.shape[0]):
        for j in range(one_pix.shape[1]):
            if one_pix[i, j] > 0:
                el = disjoint_set.get_union_by_number(one_pix[i,j])
                root = el.find()
                one_pix[i, j] = root.get_value()
    one_pix[one_pix < HIGH_THRESHOLD] = 0
    show_image(one_pix)


def show_image(img, name='image'):
    #cv2.destroyAllWindows()
    max, min = img.max(), img.min()
    img = ((img - min) / (max-min) * 255).astype(np.uint8)

    if SMALL_SCREEN:
        img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    # img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    print("shape original: {}, example[0][0]={}".format(img.shape, img[0][0]))
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #orig = grey_image.copy()
    Canny_edge(grey_image)
    #OTSU(grey_image)