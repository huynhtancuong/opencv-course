import matplotlib.pyplot as plt
import numpy as np

def julia(z, c, max_iter):
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    img = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            img[i, j] = julia(x[j] + 1j*y[i], c, max_iter)

    return img

if __name__ == '__main__':
    xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
    width, height = 500, 500
    max_iter = 100
    c = [-0.5251993 + 0.5251993j, -0.8 + 0.156j, -0.70176 - 0.3842j, -0.835 - 0.2321j, -0.8 + 0.156j]

    for ci in c:
        img = julia_set(xmin, xmax, ymin, ymax, width, height, ci, max_iter)

        plt.figure()
        plt.imshow(img, cmap='hot')
        plt.title('Julia Set with c={}'.format(ci))
        plt.colorbar()
        plt.savefig('images/julia_{}.png'.format(ci))