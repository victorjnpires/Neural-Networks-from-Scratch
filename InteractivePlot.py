# Interactive Plot functions
# Victor Jose Novaes Pires
# 2019-03-15

__version__ = '1.0'

import matplotlib.pyplot as plt
import numpy as np

blue  = [0.0000, 0.4470, 0.7410]
green = [0.4660, 0.6740, 0.1880]
red   = [0.6350, 0.0780, 0.1840]

num_classes = np.arange(10)
xticks = np.arange(0, 10, 1)
yticks = np.arange(0, 1.1, .1)

def make_plot(image, predictions, true_label, pred_label):
    plt.figure(figsize=(11, 5))

    plt.subplot(121)
    plt.title(f"Label: {true_label}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    ax = plt.subplot(122)
    bp = ax.bar(num_classes, predictions, color=blue)
    ax.set_title(f"Predicted: {pred_label}")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(0, 1.1)
    add_value_labels(ax)
    bp[pred_label].set_color(red)
    bp[true_label].set_color(green)

    plt.show()


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

        Source:
        https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        # label = "{:.1f}".format(y_value)
        # Use y_value as label and format number with two decimal places
        label = f"{y_value:.2f}"

        # Create annotation
        ax.annotate(
            label,                      # Use 'label' as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by 'space'
            textcoords="offset points", # Interpret 'xytext' as offset in points
            ha='center',                # Horizontally center label
            va=va)