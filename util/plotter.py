from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datasets.synth_datasets import gen_tasks, gen_sine_data


def visualize(representation):
    """
    Visualize the representation (weights) of a layer for omniglot (fig.5 in paper)
    :param representation: last layer's weights of RLN as numpy array
    """
    representation = representation.reshape((32, 72))
    representation = normalize(representation)
    plt.axis('off')
    pos = plt.imshow(representation, cmap="YlGn")

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.gcf().colorbar(pos, cax=cax)
    plt.show()


def plot_random_isw(n_fun_to_plot=2):
    """
    Plot some randomly generated incremental sine waves functions.
    :param n_fun_to_plot: number of functions to plot. Max 10.
    """
    tasks = gen_tasks(10)
    x_traj, y_traj, _, _ = gen_sine_data(tasks)

    for i in range(n_fun_to_plot):
        plt.scatter(x_traj[i][0][:][:, 0], y_traj[i][0][:])

    plt.grid()
    plt.show()


plot_random_isw(n_fun_to_plot=5)
