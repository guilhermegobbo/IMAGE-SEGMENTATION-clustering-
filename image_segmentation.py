# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# here we upload the image as arrays
image = plt.imread('beach.jpg')

# reshape the image to train the model (3 color channels)
image_reshaped = image.reshape(-1, 3)

# n_clusters represents the total colors we want to identify
# n_init we can say that it's for improving our cluster_centers
model = KMeans(n_clusters=5, n_init=5)
model.fit(image_reshaped)

new_img = model.cluster_centers_[model.labels_]
new_img = new_img.reshape(image.shape)

plt.imshow(new_img/255)
