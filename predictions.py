import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataget


def probmodel(model, test_images):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    return predictions, probability_model


def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if (predicted_label == true_label):
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_values_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_preds(predictions):  # Later improve to include selectable test set
    test_images, test_labels, class_names = dataget.dataget()[2:]
    nrows = 5
    ncols = 3
    nimg = nrows * ncols
    plt.figure(figsize=(2 * 2 * ncols, 2 * nrows))
    for i in range(nimg):
        plt.subplot(nrows, 2 * ncols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(nrows, 2 * ncols, 2 * i + 2)
        plot_values_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    predictions = probmodel(sys.argv[1], sys.argv[2])[0]
    plot_preds(predictions)
