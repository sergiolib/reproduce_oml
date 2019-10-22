import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def visualize(representation):
    # last_layer = rln.get_layer("Last", -1)
    # representation = last_layer.get_weights()
    representation = representation.reshape((32, 72))
    representation = normalize(representation)
    plt.axis('off')
    plt.imshow(representation)
    plt.show()