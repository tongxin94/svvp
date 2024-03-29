import numpy as np
from skimage import io
import matplotlib.pyplot as plt

img_path = '0.jpg'
plt.imshow(io.imread(img_path))
with np.load(img_path.replace(".jpg", "_vp.npz")) as npz:
    mat = npz['vp']
vp = mat / mat[2, :] * 512
vp = vp[0:2, :]
vp = vp + [[256], [256]]
plt.scatter(vp[0], vp[1])
plt.show()