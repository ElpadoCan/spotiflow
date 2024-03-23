import os
from seaborn import heatmap

import skimage.io

import numpy as np

from cellacdc.plot import imshow
from cellacdc.widgets import QDialogListbox
from cellacdc._run import _setup_app

from spotiflow.model import Spotiflow

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

app, _ = _setup_app(splashscreen=False)

selectWin = QDialogListbox('Select image', 'Selet image', os.listdir(data_path))
selectWin.exec_()

img_path = os.path.join(data_path, selectWin.selectedItemsText[0])
img_data = skimage.io.imread(img_path)

imshow(img_data)

z = input('z-slice to use?: ')
z = int(z)

img = img_data[z]

model = Spotiflow.from_pretrained("general")
points, details = model.predict(img)

imshow(img, details.heatmap, details.heatmap>0.5, points_coords=points)
import pdb; pdb.set_trace()

stack_points = []
prediction = np.zeros(img_data.shape)

for z, img in enumerate(img_data):
    points_z, details = model.predict(img)
    points = np.zeros((len(points_z), 3))
    points[:, 1:] = points_z
    points[:, 0] = z
    stack_points.append(points)
    prediction[z] = details.heatmap

stack_points = np.concatenate(stack_points)

imshow(img_data, prediction, prediction>0.5, points_coords=stack_points)

import pdb; pdb.set_trace()
