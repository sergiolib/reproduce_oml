import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

def visualize(representation):
    representation = representation.reshape((32, 72))
    representation = normalize(representation)
    plt.axis('off')
    plt.imshow(representation)
    plt.show()