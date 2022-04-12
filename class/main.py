import numpy as np
import matplotlib.pyplot as mpt
from numpy.linalg import inv

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a.transpose()
print(a)
print(b)
print(a.T)
print(inv(a))

npRandNum = np.random.rand(256, 256)
r = (255*npRandNum).astype(np.uint8)
npRandNum = np.random.rand(256, 256)
g = (255*npRandNum).astype(np.uint8)
npRandNum = np.random.rand(256, 256)
b = (255*npRandNum).astype(np.uint8)

trgb = np.array([r, g, b])
rgb = trgb.transpose((1, 2, 0))


mpt.subplot(221), mpt.imshow(r, cmap=mpt.get_cmap('gray')), mpt.title('Red->Gray Image')
mpt.subplot(222), mpt.imshow(g, cmap=mpt.get_cmap('gray')), mpt.title('Green->Gray Image')
mpt.subplot(223), mpt.imshow(b, cmap=mpt.get_cmap('gray')), mpt.title('Blue->Gray Image')
mpt.subplot(224), mpt.imshow(rgb), mpt.title('RGB Image')
mpt.show()
