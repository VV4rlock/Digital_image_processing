import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = r"image_houses.jpg"

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

    plt.plot(disp_betw)
    plt.show()
    show_image(grey_image)

def show_image(img):
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread(PATH)
    # img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    print("shape original: {}, example[0][0]={}".format(img.shape, img[0][0]))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orig = image.copy()
    grey_image = image[:, :, 2]
    OTSU(grey_image)
