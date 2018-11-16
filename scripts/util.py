import matplotlib.pyplot as plt


def load_data():
    pass


def show_image(image, interactive=False):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)

    # Only needed for non-interactive sessions
    if not interactive:
        plt.show()


def pre_process(images):
    # Scaling pixel values before feeding into the neural network
    return images / 255.0


def show_image_matrix(images, labels, names, interactive=False):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.get_cmap('binary'))
        plt.xlabel(names[labels[i]])

    # Only needed for non-interactive sessions
    if not interactive:
        plt.show()


