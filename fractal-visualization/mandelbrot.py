import matplotlib.pyplot as plt
import numpy as np

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    img = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            img[i, j] = mandelbrot(x[j] + 1j*y[i], max_iter)

    return img

if __name__ == '__main__':
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 500, 500
    max_iters = [10, 50, 100, 200, 250]

    for max_iter in max_iters:
        img = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

        plt.figure()
        plt.imshow(img, cmap='hot')
        plt.colorbar()
        plt.title('Mandelbrot Set with {} iterations'.format(max_iter))
        
        plt.savefig('images/mandelbrot_{}.png'.format(max_iter))