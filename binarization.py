import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = r"image_houses.jpg"
PRINT_HISTS = False


def OTSU(grey_image):
    show_image(grey_image)
    print("shape original: {}, example[0][0]={}".format(grey_image.shape, grey_image[0][0]))
    hist, edges = np.histogram(grey_image, range(257))
    edges = edges[:-1]
    averages = np.array([(hist[:T].dot(edges[:T]), hist[T:].dot(edges[T:])) for T in edges])
    hist = hist / hist.sum()

    f = np.array([sum(hist[:i + 1]) for i in range(len(hist))])
    doli = np.array([f[T] for T in edges])

    disp_betw = np.array([(doli[T] * (1-doli[T]) *
                           (averages[T][0]/doli[T] - averages[T][1]/(1-doli[T]))**2) for T in edges[1:-1]])

    bound = disp_betw.argmax()
    print(bound)
    grey_image[grey_image <= bound] = 0
    grey_image[grey_image > bound] = 255

    if PRINT_HISTS:
        plt.plot(edges[1:-1], f[1:-1])
        plt.plot(edges[1:-1], disp_betw/disp_betw.max())
        plt.plot(edges[1:-1], hist[1:-1]/hist[1:-1].max())
        plt.show()
    show_image(grey_image)


def get_integral_images(image: np.ndarray) -> (np.ndarray, np.ndarray):
    I = np.zeros(image.shape, dtype=np.uint64)
    I_2 = np.zeros(image.shape, dtype=np.uint64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            I[x, y] = image[x, y] + \
                                    (I[x, y - 1] if y > 0 else 0) + \
                                    (I[x - 1, y] if x > 0 else 0) - \
                                    (I[x - 1, y - 1] if x > 0 and y > 0 else 0)
            I_2[x, y] = image[x, y] ** 2 + \
                                    (I_2[x, y - 1] if y > 0 else 0) + \
                                    (I_2[x - 1, y] if x > 0 else 0) - \
                                    (I_2[x - 1, y - 1] if x > 0 and y > 0 else 0)
    return I, I_2


W = 30
k = 0.5
R = 128
def SAUVOL(grey_image: np.ndarray):
    I, I_2 = get_integral_images(grey_image)
    y_area_count = grey_image.shape[1]//W + (1 if grey_image.shape[1] % W != 0 else 0)
    for x_area in range(grey_image.shape[0]//W + (1 if grey_image.shape[0] % W != 0 else 0)):
        for y_area in range(y_area_count):
            x_beg, y_beg = x_area * W,  y_area * W
            x_end = min(x_beg + W, grey_image.shape[0]) - 1
            y_end = min(y_beg + W, grey_image.shape[1]) - 1
            S1 = I[x_end, y_end] \
                 + (I[x_beg - 1, y_beg - 1] if x_beg > 0 and y_beg > 0 else 0) \
                 - (I[x_beg - 1, y_end] if x_beg > 0 else 0)\
                 - (I[x_end, y_beg - 1] if y_beg > 0 else 0)
            S2 = I_2[x_end, y_end] \
                 + (I_2[x_beg - 1, y_beg - 1] if x_beg > 0 and y_beg > 0 else 0) \
                 - (I_2[x_beg - 1, y_end] if x_beg > 0 else 0) \
                 - (I_2[x_end, y_beg - 1] if y_beg > 0 else 0)
            n = (x_end - x_beg + 1) * (y_end - y_beg + 1)
            s = (S2 - S1 * S1 / n) / n
            t = S1/n * (1 + k * (s ** 0.5 / R - 1))
            for x in range(x_beg, x_end + 1):
                for y in range(y_beg, y_end + 1):
                    grey_image[x, y] = 255 if grey_image[x, y] > t else 0
    show_image(grey_image)


def show_image(img):
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread(PATH)
    # img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    print("shape original: {}, example[0][0]={}".format(img.shape, img[0][0]))
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #orig = grey_image.copy()
    SAUVOL(grey_image)
    #OTSU(grey_image)

