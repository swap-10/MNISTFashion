import matplotlib.pyplot as plt
import dataget


def datashape(train_images, train_labels, test_images, test_labels):
    print("Shape of training images set:", train_images.shape)
    print("Length of training labels set:", len(train_labels))
    print("Length of test images set:", test_images.shape)
    print("Length of test labels set:", len(test_labels))


def plotimg(train_images, train_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels, class_names = dataget.dataget()
    plotimg(train_images, train_labels, class_names)
    datashape(train_images, train_labels, test_images, test_labels)
