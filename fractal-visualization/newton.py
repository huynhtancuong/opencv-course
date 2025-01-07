import matplotlib.pyplot as plt
import numpy as np

def newton(z, max_iter, p, p_dot):
    '''
    Newton's method for finding roots of complex polynomials
    '''
    for n in range(max_iter):
        if abs(p(z)) < 1e-9:
            return z
        z = z - p(z) / p_dot(z)
    return z

def newton_set(xmin, xmax, ymin, ymax, width, height, max_iter, p, p_dot):
    '''
    Generate a Newton Fractal image
    '''
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    grid = np.zeros((height, width), dtype=complex)

    for i in range(height):
        for j in range(width):
            grid[i, j] = newton(x[j] + 1j*y[i], max_iter, p, p_dot)

    roots = set(grid.flatten())
    merged_roots = []
    root_dict = {}

    for root in roots:
        for merged_root in merged_roots:
            if np.isclose(root, merged_root):
                root_dict[root] = merged_root
                break
        else:
            merged_roots.append(root)
            root_dict[root] = root

    # print('Number of roots:', len(roots))
    # print('Number of merged roots:', len(merged_roots))
    # print('Roots:', root_dict)

    colors = np.linspace(0, 256, len(merged_roots))
    colors = dict(zip(merged_roots, colors))

    img = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            img[i, j] = int(colors[root_dict[grid[i, j]]])

    return img

if __name__ == '__main__':
    xmin, xmax, ymin, ymax = -10, 10, -10, 10
    width, height = 500, 500
    max_iter = 100
    p = lambda z: z**8 - 1
    p_dot = lambda z: 8*z**7

    img = newton_set(xmin, xmax, ymin, ymax, width, height, max_iter, p, p_dot)

    plt.imshow(img, cmap='tab20')
    plt.colorbar()
    plt.title('Newton Fractal with p(z) = z^8 - 1')
    # plt.show()
    plt.savefig('images/newton_2.png')