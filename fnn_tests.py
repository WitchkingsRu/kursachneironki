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


def create_wide_model(input_shape, num_classes, width_multiplier=1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(256 * width_multiplier, activation='relu'))
    model.add(layers.Dense(128 * width_multiplier, activation='relu'))
    model.add(layers.Dense(64 * width_multiplier, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_deep_model(input_shape, num_classes, depth_multiplier=1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for _ in range(1 * depth_multiplier):
        model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_dense_models():
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_and_prepare_data()

    results = {
        'wide_models': [],
        'deep_models': []
    }

    # Эксперименты с шириной
    print("Эксперименты с шириной сети:")
    for width in [1, 2, 4, 8]:
        model = create_wide_model(train_images.shape[1:], 10, width)
        history = model.fit(train_images, train_labels, epochs=20, batch_size=128,
                            validation_data=(val_images, val_labels), verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        results['wide_models'].append({
            'width': width,
            'test_accuracy': test_acc,
            'params': model.count_params()
        })
        print(f"Width: {width}, Accuracy: {test_acc:.4f}, Params: {model.count_params()}")

    # Эксперименты с глубиной
    print("\nЭксперименты с глубиной сети:")
    for depth in [1, 2, 4, 8]:
        model = create_deep_model(train_images.shape[1:], 10, depth)
        history = model.fit(train_images, train_labels, epochs=20, batch_size=128,
                            validation_data=(val_images, val_labels), verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        results['deep_models'].append({
            'depth': depth,
            'test_accuracy': test_acc,
            'params': model.count_params()
        })
        print(f"Depth: {depth}, Accuracy: {test_acc:.4f}, Params: {model.count_params()}")

    return results


def visualize_dense_results(results):
    plt.figure(figsize=(12, 5))

    # Графики для широких моделей
    plt.subplot(1, 2, 1)
    widths = [x['width'] for x in results['wide_models']]
    accuracies = [x['test_accuracy'] for x in results['wide_models']]
    plt.plot(widths, accuracies, 'o-')
    plt.xlabel('Width multiplier')
    plt.ylabel('Test accuracy')
    plt.title('Влияние ширины сети на точность')

    # Графики для глубоких моделей
    plt.subplot(1, 2, 2)
    depths = [x['depth'] for x in results['deep_models']]
    accuracies = [x['test_accuracy'] for x in results['deep_models']]
    plt.plot(depths, accuracies, 'o-')
    plt.xlabel('Depth multiplier')
    plt.ylabel('Test accuracy')
    plt.title('Влияние глубины сети на точность')

    plt.tight_layout()
    plt.savefig('dense_networks_results.png')
    plt.show()

    # Сводка
    print("\nСводка по полносвязным сетям:")
    print("Лучшая широкая модель:", max(results['wide_models'], key=lambda x: x['test_accuracy']))
    print("Лучшая глубокая модель:", max(results['deep_models'], key=lambda x: x['test_accuracy']))


if __name__ == "__main__":
    dense_results = train_and_evaluate_dense_models()
    visualize_dense_results(dense_results)
    with open('dense_results.json', 'w') as f:
        json.dump(dense_results, f, indent=2)
