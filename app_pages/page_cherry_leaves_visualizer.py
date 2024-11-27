import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
import itertools

def page_cherry_leaves_visualizer_body():
    st.write("### Cherry Leaves Visualiser")
    st.info(
        f"A study that visually differentiates between a healthy and powdery "
        f"mildew cherry leaf."
    )
    st.success(
        f"A healthy cherry leaf and a powdery mildew-infected cherry leaf can be distinguished by:\n\n"
        f"**Healthy Cherry Leaf:**\n"
        f"- Color: Even green hue.\n"
        f"- Texture: Smooth surface with no abnormal growths or spots.\n"
        f"- Shape: Maintains its normal, undistorted form.\n\n"
        f"**Powdery Mildew-Infected Cherry Leaf:**\n"
        f"- Color: Exhibits white or grayish powdery patches, typically starting on the upper surface.\n"
        f"- Texture: Powdery or dusty appearance on the leaf.\n"
        f"- Shape: Leaves may be curled, twisted, or distorted, with severe cases leading to yellowing "
        f"and early leaf drop.\n"
    )

    version = 'v1'

    # Difference Between Average and Variability Images
    if st.checkbox("Difference between average and variability image"):
        avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            f"* We notice the average and variability images did not show "
            f"patterns where we could intuitively differentiate one from another. "
            f"However, a small difference in the color pigment of the average images is seen for both labels."
        )
        st.image(avg_powdery_mildew, caption="Powdery Mildew Leaves - Average and Variability")
        st.image(avg_healthy, caption="Healthy Leaves - Average and Variability")
        st.write("---")

    # Differences Between Average Images
    if st.checkbox("Differences between a healthy and powdery mildew leaf"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")
        st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate one from another."
        )
        st.image(diff_between_avgs, caption="Difference between average images")

    # Image Montage
    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = "inputs/cherry_leaves_dataset/cherry-leaves"
        labels = os.listdir(my_data_dir + "/validation")

        label_to_display = st.selectbox(
            label="Select Leaf Category (Healthy or Powdery Mildew)",
            options=labels,
            index=0,
        )

        if st.button("Create Montage"):
            image_montage(
                dir_path=my_data_dir + "/validation",
                label_to_display=label_to_display,
                nrows=8,
                ncols=3,
                figsize=(10, 25),
            )
        st.write("---")

    st.write(
        f"For additional information, please visit the "
        f"[Project README file](https://github.com/Sammy92dec/mildew-detection-in-cherry-leaves/blob/main/README.md)."
    )


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    """
    Display a montage of images from a specified directory
    """
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # Subset the class you are interested in displaying
    if label_to_display in labels:
        images_list = os.listdir(dir_path + "/" + label_to_display)
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Decrease nrows or ncols to create your montage.\n"
                f"There are {len(images_list)} images in your subset. "
                f"You requested a montage with {nrows * ncols} spaces."
            )
            return

        # Create list of axes indices based on nrows and ncols
        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        # Create a Figure and display images
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, nrows * ncols):
            img = imread(dir_path + "/" + label_to_display + "/" + img_idx[x])
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                f"Width {img_shape[1]}px x Height {img_shape[0]}px"
            )
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])

        plt.tight_layout()
        st.pyplot(fig=fig)

    else:
        st.warning("The label you selected doesn't exist.")
        st.write(f"The existing options are: {labels}")
