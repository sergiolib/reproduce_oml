import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def visualize(representation):
    """
    Visualize the representation (weights) of a layer for omniglot (fig.5 in paper)
    :param representation: last layer's weights of RLN as numpy array
    """
    representation = representation.reshape((32, 72))
    representation = normalize(representation)
    plt.axis('off')
    plt.imshow(representation)
    plt.show()
