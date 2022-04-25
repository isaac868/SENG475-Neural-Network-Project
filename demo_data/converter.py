import numpy as np
import sys
from PIL import Image

img = np.asarray(Image.open(str(sys.argv[1])).convert("L").resize((28,28)))

st = ""
for row in img:
    for val in row:
        st += str(255 - val)
        st += ","
print st