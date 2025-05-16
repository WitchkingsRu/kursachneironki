import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import json

# Настройки для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)


def load_and_prepare_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def create_conv_model(input_shape, num_classes, depth=3, width=32):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(depth):
        model.add(layers.Conv2D(width * (2 ** i), (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_conv_models():
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_and_prepare_data()

    results = {
        'conv_models_depth': [],
        'conv_models_width': []
    }

    # Эксперименты с глубиной
    print("Эксперименты с глубиной сверточной сети:")
    for depth in [2, 3, 4, 5]:
        model = create_conv_model(train_images.shape[1:], 10, depth=depth, width=32)
        history = model.fit(train_images, train_labels, epochs=20, batch_size=128,
                            validation_data=(val_images, val_labels), verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        results['conv_models_depth'].append({
            'depth': depth,
            'test_accuracy': test_acc,
            'params': model.count_params()
        })
        print(f"Depth: {depth}, Accuracy: {test_acc:.4f}, Params: {model.count_params()}")

    # Эксперименты с шириной
    print("\nЭксперименты с шириной сверточной сети:")
    for width in [16, 32, 64]:
        model = create_conv_model(train_images.shape[1:], 10, depth=3, width=width)
        history = model.fit(train_images, train_labels, epochs=20, batch_size=128,
                            validation_data=(val_images, val_labels), verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        results['conv_models_width'].append({
            'width': width,
            'test_accuracy': test_acc,
            'params': model.count_params()
        })
        print(f"Width: {width}, Accuracy: {test_acc:.4f}, Params: {model.count_params()}")

    return results


def visualize_conv_results(results):
    plt.figure(figsize=(12, 5))

    # Графики для глубины
    plt.subplot(1, 2, 1)
    depths = [x['depth'] for x in results['conv_models_depth']]
    accuracies = [x['test_accuracy'] for x in results['conv_models_depth']]
    plt.plot(depths, accuracies, 'o-')
    plt.xlabel('Conv depth')
    plt.ylabel('Test accuracy')
    plt.title('Влияние глубины сверточной сети на точность')

    # Графики для ширины
    plt.subplot(1, 2, 2)
    widths = [x['width'] for x in results['conv_models_width']]
    accuracies = [x['test_accuracy'] for x in results['conv_models_width']]
    plt.plot(widths, accuracies, 'o-')
    plt.xlabel('Conv width')
    plt.ylabel('Test accuracy')
    plt.title('Влияние ширины сверточной сети на точность')

    plt.tight_layout()
    plt.savefig('conv_networks_results.png')
    plt.show()


    # Сводка
    print("\nСводка по сверточным сетям:")
    print("Лучшая модель по глубине:", max(results['conv_models_depth'], key=lambda x: x['test_accuracy']))
    print("Лучшая модель по ширине:", max(results['conv_models_width'], key=lambda x: x['test_accuracy']))

if __name__ == "__main__":
    conv_results = train_and_evaluate_conv_models()
    visualize_conv_results(conv_results)
    with open('conv_results.json', 'w') as f:
        json.dump(conv_results, f, indent=2)
