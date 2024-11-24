import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Label Frequencies on Train, Validation and Test Sets")

    st.info(
        f" The cherry leaves dataset was divided into three subsets:\n\n"
        f" * The training set comprises 1,472 images, representing 70% of the entire dataset. This data is used to "
        f" train the model, enabling it to generalize and make predictions on new, unseen data.\n\n"
        f" * The validation set comprises 210 images, representing 10% of the entire dataset. Assists in enhancing the "
        f" model's performance by refining it after each epoch, which is a full pass of the training set through the model.\n\n"
        f" * The test set comprises 422 images, representing 20% of the entire dataset. Provides information about the model's "
        f" final accuracy after the training phase is completed. This evaluation uses a batch of data that the model has never seen before.")

    labels_distribution = plt.imread(f"outputs/{version}/label_dist.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")

    st.info(
        f" **Model training - Accuracy and Loss**\n\n"
        f" Accuracy measures how closely the model's predictions (accuracy) match the true data (val_acc).\n"
        f" A good model that performs well on unseen data demonstrates its ability to generalize and avoid overfitting to the training dataset.\n\n"
        f" The loss is the total of errors made for each example in the training (loss) or validation (val_loss) sets.\n"
        f" The loss value indicates how poorly or well a model performs after each optimization iteration.")

    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_accuracy.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')

    st.info(
        f" **The model learning ROC curve**\n\n"
        f" Loss (Blue) and Validation Loss (Green):\n "
        f" * Loss measures the prediction accuracy, the lower the loss the better.\n"
        f" indicating good performance on unseen data.\n\n"
        f"Accuracy (Orange) and Validation Accuracy (Red):\n"
        f" * Accuracy measures the proportion of correct predictions.\n\n"
        f"In summary, both the training and validation losses decrease and stabilize at low values, indicating "
        f" that the model has converged; the high training and validation accuracies show the model is "
        f" performing well; and the close alignment between the training and validation metrics indicates the "
        f" model is not overfitting and can generalize well on new data.")

    model_results_curve = plt.imread(f"outputs/{version}/model_combined.png")
    st.image(model_results_curve, caption='Model Training Accuracy & Losses')

    st.info(
        f" **Confusion Matrix**\n\n"
        f" The confusion matrix is used to evaluate the performance of the model."
        f" It compares the actual labels (true values) with the predicted labels given by the model.\n\n"
        f" * The model correctly identified 210 healthy instances and 209 powdery mildew instances.\n "
        f" * The model made 3 mistakes: 1 healthy instance was incorrectly classified as powdery mildew, and 2 "
        f" powdery mildew instances were incorrectly classified as healthy.")

    conf_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(conf_matrix, caption='Confusion matrix')

    st.info(
        f" **Classification Report**\n\n"
        f" The report provides a detailed performance analysis of the model."
        f" It includes various metrics for evaluating the accuracy and effectiveness of the model.\n\n"
        f" * In summary, the model performs exceptionally well, with high precision, recall, and F1-scores for both classes, "
        f" and an overall accuracy of 99%.")

    class_report = plt.imread(f"outputs/{version}/clf_report.png")
    st.image(class_report, caption='Classification Report')
    st.write("---")

    st.write("### Generalised Performance on Test Set")

    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    st.info(
        f" The model has achieved an accuracy of 99%")
    
    st.write(
        f"For additional information, please visit "
        f"[Project README file](https://github.com/Sammy92dec/mildew-detection-in-cherry-leaves/"
        f"blob/main/README.md)."
    )