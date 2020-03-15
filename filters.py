import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

#doi:10.1016/j.sigpro.2009.03.025
#https://www.hindawi.com/journals/tswj/2014/230425/

IMAGE_PATH = r"DIP/planes_forg_2.jpg"
THETA = 10 / 180 * np.pi
DEBUG = False
SMALL_SCREEN = True
#DEBUG1 = False

gauss_5 = signal.gaussian(5, 1).reshape(-1, 1)
gauss_filter = np.dot(gauss_5, gauss_5.T)
gauss_filter = gauss_filter / (gauss_filter.shape[0] * gauss_filter.shape[1])

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


def preprocessing(image) -> np.ndarray:
    logger.info("preprocessing")
    #image = apply_filter(image, gauss_filter)
    y = apply_filter(image, SOBEL_Y).astype(np.float32)
    x = apply_filter(image, SOBEL_X).astype(np.float32)
    x[x == 0] = 0.0000001
    #print(x.min(), y.min())
    tan = y / x

    #print(tan.max())
    #print(np.tan(np.pi/2), np.tan(np.pi/2 - THETA), np.tan(THETA), np.tan(0))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[tan >= np.tan(np.pi/2 - THETA)] = 1
    mask[tan <= np.tan(THETA)] = 1
    #mask = mask

    #print(second_derivative)
    #show_image(apply_filter(image, derivate_filter))
    #show_image(mask)
    #show_image(y)
    #show_image(x)
    return mask


def extract_horizontal(image: np.ndarray, mask) -> np.ndarray:
    logger.info("extract horizontal")
    image_padded = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_REPLICATE).astype(np.int32)
    d = np.zeros(image_padded.shape, dtype=np.uint8)
    for row in range(16, image_padded.shape[0] - 16):
        d[row, :] = np.abs(2 * image_padded[row, :] - image_padded[row - 1, :] - image_padded[row + 1, :])
    if DEBUG:
        show_image(d, 'second horizontal derivative')
    d[16:d.shape[0] - 16, 16:d.shape[1] - 16] = d[16:d.shape[0] - 16, 16:d.shape[1] - 16] * mask

    es = np.zeros(image_padded.shape, dtype=np.uint32)
    for x in range(16, image_padded.shape[1] - 16):
        ac = d[:, x - 16]
        for i in range(-15, 17, 1):
            ac += d[:, x + i]
        es[:, x] = ac

    logging.info("\tcomputing e")
    e = np.zeros(image_padded.shape, dtype=np.int32)
    for row in range(16, image_padded.shape[0] - 16):
        t = es[row - 16: row + 17, :]
        e[row, :] = es[row, :] - np.array(list( (np.median(i) for i in t.T) ))

    #e = np.abs(e)
    if DEBUG:
        e_show = e - e.min()
        e_show = (e_show / e_show.max() * 255).astype(np.uint8)
        show_image(e_show, 'mid e horizontal')

    logging.info("\tcomputing gh")
    gh = np.zeros(image_padded.shape, dtype=np.int32)
    for row in range(16, image_padded.shape[0] - 16):
        t = np.array([e[i, :] for i in [row-16, row-8, row, row+8, row+16]])
        gh[row, :] = np.array(list((np.median(i) for i in t.T)))

    #gh = np.abs(gh)
    if DEBUG:
        gh_show = gh - gh.min()
        gh_show = (gh_show / gh_show.max() * 255).astype(np.uint8)
        show_image(gh_show, 'extracted horizontal')

    #hist, edges = np.histogram(grey_image, range(257))
    #plt.plot(hist)
    #plt.show()

    return gh[16:gh.shape[0] - 16 - (gh.shape[0] - 16) % 8, 16:gh.shape[1] - 16 - (gh.shape[1] - 16) % 8]

def extract_vertical(image, mask) -> np.ndarray:
    logger.info("extract vertical")
    image_padded = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_REPLICATE).astype(np.int32)

    d = np.zeros(image_padded.shape, dtype=np.uint8)
    for column in range(16, image_padded.shape[1] - 16):
       d[:, column] = np.abs(2 * image_padded[:, column] - image_padded[:, (column - 1)] - image_padded[:, (column + 1)])
    if DEBUG:
        show_image(d, 'verdical derivate')
    d[16:d.shape[0]-16, 16:d.shape[1]-16] = d[16:d.shape[0]-16, 16:d.shape[1]-16] * mask
    es = np.zeros(image_padded.shape, dtype=np.uint32)
    for y in range(16, image_padded.shape[0] - 16):
        ac = d[y - 16, :]
        for i in range(-15, 17, 1):
            ac += d[y + i, :]
        es[y, :] = ac

    logger.info("\tcomputing e")
    e = np.zeros(image_padded.shape, dtype=np.int32)
    for column in range(16, image_padded.shape[1] - 16):
        t = es[:, column - 16: column + 17]
        e[:, column] = es[:, column] - np.array(list((np.median(i) for i in t))) #.T?
    if DEBUG:
        e_show = e - e.min()
        e_show = (e_show / e_show.max() * 255).astype(np.uint8)
        show_image(e_show, 'mid e vertical')

    logger.info("\tcomputing gv")
    gv = np.zeros(image_padded.shape, dtype=np.int32)
    for column in range(16, image_padded.shape[1] - 16):
        t = np.array([e[:, i] for i in [column - 16, column - 8, column, column + 8, column + 16]])
        gv[:, column] = np.array(list((np.median(i) for i in t.T)))

    if DEBUG:
        gv_show = gv - gv.min()
        gv_show = (gv_show / gv_show.max() * 255).astype(np.uint8)
        show_image(gv_show, 'extracted vertical')
    return gv[16:gv.shape[0] - 16 - (gv.shape[0] - 16) % 8, 16:gv.shape[1] - 16 - (gv.shape[1] - 16) % 8]


def find_forgery(image):
    mask = preprocessing(image)
    #image = image * mask # непонятно куда применять маску, т к исключение пикселей добавляет новые градиенты
    gv = extract_vertical(image, mask)
    gh = extract_horizontal(image, mask)
    g = gh + gv
    logger.info("BAG extracted")
    #g[g < 0] = 0

    if True:
        show_g = g - g.min()
        show_g = (show_g / show_g.max() * 255).astype(np.uint8)
        show_image(show_g, "BAG image")
    for y in range(0, g.shape[0], 8):
        for x in range(0, g.shape[1], 8):
            block = g[y:y+8, x:x+8]
            f1 = max([block[1:7, i].sum() for i in range(1, 7)])
            f2 = min([block[1:7, i].sum() for i in [0, 7]])
            f3 = max([block[i, 1:7].sum() for i in range(1, 7)])
            f4 = min([block[i, 1:7].sum() for i in [0, 7]])
            g[y:y + 8, x:x + 8] = f1 - f2 + f3 - f4
    logger.info("finish")
    #print(g.min(), g.max())

    show_g = g - g.min()
    show_g = (show_g / show_g.max() * 255).astype(np.uint8)
    show_image(image, "orig")
    show_image(show_g, "result")




def show_image(img, name='image'):
    #cv2.destroyAllWindows()

    if SMALL_SCREEN:
        img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    #show_image(img, 'orig')
    #img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
    print("shape original: {}, example[0][0]={}".format(img.shape, img[0][0]))
    #grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    #show_image(grey_image)
    find_forgery(grey_image)
    #show_image(apply_filter(grey_image, derivate_filter))
    #show_image(grey_image)


