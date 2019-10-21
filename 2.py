import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = r"DIP/2/image20.png"
N = 4
P = 0.3
PRINT_HISTS = False
spread_for_all = True

def show_image(img):
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_CDF_by_image(grey_image):
    hist, edges = np.histogram(grey_image, range(257))
    edges = edges[:-1]
    # show_image(grey_image)
    m = hist.max()
    bound = int(m * P)
    mass = 0
    above_zero = []
    for i in range(256):
        if hist[i] > bound:
            mass += hist[i] - bound
            hist[i] = bound
        if hist[i]>0:
            above_zero.append(i)
    if spread_for_all:
        additional = mass // 256
        for i in range(256):
            hist[i] += additional
        for j in range(mass % 256):
            hist[j] += 1
    else:
        additional = mass // len(above_zero)
        for i in above_zero:
            hist[i] += additional
        for j in above_zero[:mass % len(above_zero)]:
            hist[j] += 1

    hist = hist / hist.sum()

    f = np.array([sum(hist[:i + 1]) for i in range(len(hist))]) * 255
    return edges, f.astype(np.uint8)


def apply_equalization_to_sub_image_CR(image, x, y, height, width, hist):
    #print(x,y,width,height)
    for _x in range(x, x + height):
        for _y in range(y, y + width):
            image[_x][_y] = hist[1][image[_x][_y]]
    #image[x:x+height, y:y+width] = hist[1][100]


def apply_equalization_to_sub_image_BR_vertical(image, x, y, height, width, hist_t, hist_b):
    for _x in range(x, x + height):
        a = (_x - x) / height
        for _y in range(y, y + width):
            image[_x][_y] = min(255, (1 - a) * hist_t[1][image[_x][_y]] + a * hist_b[1][image[_x][_y]])


def apply_equalization_to_sub_image_BR_horizontal(image, x, y, height, width, hist_l, hist_r):
    for _y in range(y, y + width):
        a = (_y - y) / width
        for _x in range(x, x + height):
            image[_x][_y] = min(255, (1 - a) * hist_l[1][image[_x][_y]] + (a) * hist_r[1][image[_x][_y]])


cache = [[]]
def gen_influence_map(height, width):
    global cache
    if height == len(cache) and width == len(cache[0]):
        #print('hit')
        return cache
    cache = []
    for a in np.linspace(0,1,height):
        tmp = []
        for b in np.linspace(0,1,width):
            tmp.append(((1-a)*(1-b), (1-a)*b, a * (1 - b), a * b))
        cache.append(tmp)
    return cache


def apply_equalization_to_sub_image_IR(image, x, y, height, width, hist1, hist2, hist3, hist4): # 1 2
    #print(x, y, width, height)                                                                 # 3 4
    influence_map = gen_influence_map(height, width)
    for _x in range(height):
        for _y in range(width):
            image[x + _x][y + _y] = min(255, influence_map[_x][_y][0] * hist1[1][image[x + _x][y + _y]] + \
                            influence_map[_x][_y][1] * hist2[1][image[x + _x][y + _y]] + \
                            influence_map[_x][_y][2] * hist3[1][image[x + _x][y + _y]] + \
                            influence_map[_x][_y][3] * hist4[1][image[x + _x][y + _y]])


def CLAHE(grey_image):
    orig = grey_image.copy()
    subimage_height, subimage_width = grey_image.shape[0] // N, grey_image.shape[1] // N
    print(subimage_width, subimage_height)
    hists = []
    for i in range(N):
        tmp = []
        for j in range(N):
            subimage = grey_image[i*subimage_height : i*subimage_height + min(subimage_height,
                                                          grey_image.shape[0] - i*subimage_height),
                       j*subimage_width : j*subimage_width + min(subimage_width,
                                              grey_image.shape[1] - j*subimage_width)]
            #print(subimage)
            tmp.append(get_CDF_by_image(subimage))
        hists.append(tmp)

    #test_image = np.zeros(grey_image.shape, dtype=np.uint8)
    angels = [
        (0, 0, subimage_height // 2, subimage_width // 2, hists[0][0]),
        (0, subimage_width * (N - 1) + subimage_width // 2, subimage_height // 2, subimage_width // 2, hists[0][-1]),
        (subimage_height * (N - 1) + subimage_height // 2, 0, subimage_height // 2, subimage_width // 2, hists[-1][0]),
        (subimage_height * (N - 1) + subimage_height // 2, subimage_width * (N - 1) + subimage_width // 2, subimage_height // 2, subimage_width // 2, hists[-1][-1]),
    ]
    boarder_vertical = [
        [(i*subimage_height + subimage_height // 2, 0, subimage_height, subimage_width // 2, hists[i][0],
          hists[i+1][0]) for i in range(N-1)],
        [(i * subimage_height + subimage_height // 2, subimage_width * (N - 1) + subimage_width // 2,
          subimage_height, subimage_width // 2, hists[i][-1], hists[i + 1][-1]) for i in range(N - 1)],
    ]
    boarder_horizontal = [
        [(0, i * subimage_width + subimage_width // 2, subimage_height // 2, subimage_width , hists[0][i],
          hists[0][i + 1]) for i in range(N - 1)],
        [(subimage_height * (N - 1) + subimage_height // 2, i * subimage_width + subimage_width // 2,
          subimage_height // 2, subimage_width , hists[-1][i], hists[-1][i + 1]) for i in range(N - 1)],
    ]
    for boarder in boarder_vertical:
        for title in boarder:
            apply_equalization_to_sub_image_BR_vertical(grey_image, *title)
    for boarder in boarder_horizontal:
        for title in boarder:
            apply_equalization_to_sub_image_BR_horizontal(grey_image, *title)
    for i in angels:
        apply_equalization_to_sub_image_CR(grey_image, *i)

    for i in range(N - 1):
        for j in range(N - 1):
            apply_equalization_to_sub_image_IR(grey_image, i * subimage_height + subimage_height // 2,
                                               j * subimage_width + subimage_width // 2, subimage_height, subimage_width
                                               , hists[i][j], hists[i][j + 1], hists[i + 1][j], hists[i + 1][j + 1])
    show_image(orig)
    show_image(grey_image)
    if PRINT_HISTS:
        fig = plt.figure()
        for i in range(N):
            for j in range(N):
                a = fig.add_subplot(N, N, i*N + j + 1)
                a.plot(*hists[i][j])
        plt.show(fig)



if __name__ == "__main__":
    img = cv2.imread(PATH)
    #img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    print("shape original: {}, example[0][0]={}".format(img.shape, img[0][0]))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orig = image.copy()
    grey_image = image[:, :, 2] # only V

    CLAHE(grey_image)
    show_image(cv2.cvtColor(orig, cv2.COLOR_HSV2BGR))
    show_image(cv2.cvtColor(image, cv2.COLOR_HSV2BGR))



