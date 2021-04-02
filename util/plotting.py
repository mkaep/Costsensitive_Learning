"""

@author: Martin Käppel
"""

import seaborn as sb
import os
from matplotlib import pyplot


def plot_barchart_from_dictionary(dictionary, title, xlabel, ylabel, save=False, output_file=None, encode=False,
                                  encoder=None):
    sb.set_theme(style="darkgrid")
    if not encode:
        ax = sb.barplot(x=list(dictionary.keys()), y=list(dictionary.values()))
    else:
        if encoder is None:
            raise Exception("You must specify the encoder!")
        else:
            ax = sb.barplot(x=encoder.transform(list(dictionary.keys())), y=list(dictionary.values()))

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set(ylim=(0, max(dictionary.values()) * 1.2))
    pyplot.xticks(rotation=90)
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = str(int(y_value))

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',  # Horizontally center label
            va=va,  # Vertically align label differently for positive and negative values.
            fontsize=10)
    if save:
        pyplot.tight_layout()
        if output_file is None:
            raise Exception("Error: if save option true you have to specify a output file!")
        pyplot.savefig(os.path.join(output_file, title + ".png"))

    pyplot.show()
