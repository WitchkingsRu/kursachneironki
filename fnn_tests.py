import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import json

# Настройки для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)


def load_and_prepare_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Нормализация с учетом среднего и std (лучше чем просто /255)
    mean = np.mean(train_images, axis=(0, 1, 2))
    std = np.std(train_images, axis=(0, 1, 2))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def create_improved_wide_model(input_shape, num_classes, width_multiplier=1):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(512 * width_multiplier, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256 * width_multiplier, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128 * width_multiplier, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_improved_deep_model(input_shape, num_classes, depth_multiplier=1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    for i in range(2 * depth_multiplier):
        model.add(layers.Dense(256 if i < depth_multiplier else 128,
                               activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        if i % 2 == 0:
            model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_evaluate_dense_models():
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_and_prepare_data()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    results = {
        'wide_models': [],
        'deep_models': []
    }

    # Эксперименты с шириной
    print("Эксперименты с шириной сети:")
    for width in [1, 2, 3]:
        model = create_improved_wide_model(train_images.shape[1:], 10, width)
        history = model.fit(train_images, train_labels,
                            epochs=100,
                            batch_size=128,
                            validation_data=(val_images, val_labels),
                            callbacks=callbacks,
                            verbose=1)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        results['wide_models'].append({
            'width': width,
            'test_accuracy': test_acc,
            'params': model.count_params(),
            'epochs': len(history.history['val_accuracy'])
        })
        print(f"Width: {width}, Accuracy: {test_acc:.4f}, Params: {model.count_params()}")

    # Эксперименты с глубиной
    print("\nЭксперименты с глубиной сети:")
    for depth in [2, 3, 4]:
        model = create_improved_deep_model(train_images.shape[1:], 10, depth)
        history = model.fit(train_images, train_labels,
                            epochs=100,
                            batch_size=128,
                            validation_data=(val_images, val_labels),
                            callbacks=callbacks,
                            verbose=1)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        results['deep_models'].append({
            'depth': depth,
            'test_accuracy': test_acc,
            'params': model.count_params(),
            'epochs': len(history.history['val_accuracy'])
        })
        print(f"Depth: {depth}, Accuracy: {test_acc:.4f}, Params: {model.count_params()}")

    return results


def visualize_dense_results(results):
    plt.figure(figsize=(14, 6))

    # Графики точности
    plt.subplot(1, 2, 1)
    widths = [x['width'] for x in results['wide_models']]
    accuracies = [x['test_accuracy'] for x in results['wide_models']]
    plt.plot(widths, accuracies, 'o-', label='Wide models')

    depths = [x['depth'] for x in results['deep_models']]
    accuracies = [x['test_accuracy'] for x in results['deep_models']]
    plt.plot(depths, accuracies, 'o-', label='Deep models')

    plt.xlabel('Width/Depth multiplier')
    plt.ylabel('Test accuracy')
    plt.title('Сравнение точности широких и глубоких моделей')
    plt.legend()

    # Графики параметров
    plt.subplot(1, 2, 2)
    params = [x['params'] for x in results['wide_models']]
    plt.plot(widths, params, 'o-', label='Wide models')

    params = [x['params'] for x in results['deep_models']]
    plt.plot(depths, params, 'o-', label='Deep models')

    plt.xlabel('Width/Depth multiplier')
    plt.ylabel('Number of parameters')
    plt.title('Количество параметров в моделях')
    plt.legend()
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('improved_dense_results.png')
    plt.show()

    # Сводка
    print("\nСводка по улучшенным полносвязным сетям:")
    best_wide = max(results['wide_models'], key=lambda x: x['test_accuracy'])
    best_deep = max(results['deep_models'], key=lambda x: x['test_accuracy'])

    print(f"Лучшая широкая модель: Width={best_wide['width']}, "
          f"Accuracy={best_wide['test_accuracy']:.4f}, "
          f"Params={best_wide['params']}, Epochs={best_wide['epochs']}")

    print(f"Лучшая глубокая модель: Depth={best_deep['depth']}, "
          f"Accuracy={best_deep['test_accuracy']:.4f}, "
          f"Params={best_deep['params']}, Epochs={best_deep['epochs']}")


if __name__ == "__main__":
    dense_results = train_and_evaluate_dense_models()
    visualize_dense_results(dense_results)
    with open('improved_dense_results.json', 'w') as f:
        json.dump(dense_results, f, indent=2)