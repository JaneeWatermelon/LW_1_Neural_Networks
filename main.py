from math import exp, tanh, sqrt
from typing import Callable, List, Tuple, Dict
import random
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relu(x: float) -> float:
    """ReLU-активация."""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Производная ReLU."""
    return 1.0 if x > 0 else 0.0


def sigmoid(x: float) -> float:
    """Сигмоида с защитой от переполнения."""
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    z = exp(x)
    return z / (1 + z)


def sigmoid_derivative(x: float) -> float:
    """Производная сигмоиды."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_activation(x: float) -> float:
    """Гиперболический тангенс."""
    return tanh(x)


def tanh_derivative(x: float) -> float:
    """Производная tanh."""
    return 1 - tanh(x) ** 2


def softmax(x: List[float]) -> List[float]:
    """
    Softmax-функция для многоклассовой классификации.
    """
    max_x = max(x)
    exp_x = [exp(v - max_x) for v in x]
    s = sum(exp_x)
    return [v / s for v in exp_x]


class Neuron:
    """
    Формальный нейрон с поддержкой backpropagation.
    """

    def __init__(self, num_inputs: int, activation: str = "relu") -> None:
        limit = (1.0 / num_inputs) ** 0.5
        self.weights = [random.uniform(-limit, limit) for _ in range(num_inputs)]
        self.bias = random.uniform(-limit, limit)
        self.num_inputs = num_inputs

        if activation == "relu":
            self.activation = relu
            self.activation_deriv = relu_derivative
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_deriv = sigmoid_derivative
        elif activation == "tanh":
            self.activation = tanh_activation
            self.activation_deriv = tanh_derivative
        elif activation == "linear":
            self.activation = lambda x: x
            self.activation_deriv = lambda x: 1.0
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.X = None
        self.z = None
        self.output = None
        self.delta = None

        self.grad_weights = [0.0] * num_inputs
        self.grad_bias = 0.0

    
    def forward(self, X: List[float]) -> float:
        """Прямой проход нейрона."""
        self.X = X
        self.z = sum(w * x for w, x in zip(self.weights, X)) + self.bias
        self.output = self.activation(self.z)
        return self.output

    
    def backward(self, error: float) -> List[float]:
        """Обратное распространение ошибки."""
        self.delta = error * self.activation_deriv(self.z)

        for i, x in enumerate(self.X):
            self.grad_weights[i] += self.delta * x
        self.grad_bias += self.delta

        return [self.delta * w for w in self.weights]

    
    def update(self, learning_rate: float) -> None:
        """Обновление параметров нейрона."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.grad_weights[i]
        self.bias -= learning_rate * self.grad_bias


class DenseLayer:
    """
    Полносвязный слой нейронов.
    """

    def __init__(self, num_neurons: int, num_inputs: int, activation: str) -> None:
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_neurons)]
        self.X = None

    
    def forward(self, inputs: List[float]) -> List[float]:
        """Прямой проход слоя."""
        self.X = inputs
        return [neuron.forward(inputs) for neuron in self.neurons]

    
    def backward(self, errors: List[float]) -> List[float]:
        """Backpropagation через слой."""
        prev_errors = [0.0] * len(self.X)

        for error, neuron in zip(errors, self.neurons):
            neuron_errors = neuron.backward(error)
            for i in range(len(prev_errors)):
                prev_errors[i] += neuron_errors[i]

        return prev_errors
    
    def update(self, learning_rate: float) -> None:
        """Обновление параметров слоя."""
        for neuron in self.neurons:
            neuron.update(learning_rate)
            neuron.grad_weights = [0.0] * len(neuron.grad_weights)
            neuron.grad_bias = 0.0


class NeuralNetwork:
    """
    Полносвязная нейронная сеть для многоклассовой классификации.
    """

    def __init__(self) -> None:
        self.layers = []
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    
    def add_layer(self, layer: DenseLayer) -> None:
        """Добавление слоя в сеть."""
        self.layers.append(layer)

    
    def forward(self, inputs: List[float]) -> List[float]:
        """Прямой проход по сети."""
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    
    def predict(self, inputs: List[float]) -> tuple[int, List[float]]:
        """Предсказание класса."""
        logits = self.forward(inputs)
        probs = softmax(logits)
        return probs.index(max(probs)), probs

    
    def compute_loss(self, outputs: List[float], target: int) -> float:
        """Cross-entropy loss."""
        probs = softmax(outputs)
        return -np.log(probs[target] + 1e-8)

    
    def compute_gradients(self, outputs: List[float], target: int) -> List[float]:
        """Градиенты softmax + cross-entropy."""
        probs = softmax(outputs)
        probs[target] -= 1.0
        return probs

    
    def train_on_batch(self, X_batch, y_batch, learning_rate):
        """Обучение на одном батче"""
        batch_loss = 0.0
        correct = 0
        
        for X, y in zip(X_batch, y_batch):
            # Прямой проход
            outputs = self.forward(X)
            
            # Вычисление потерь
            loss = self.compute_loss(outputs, y)
            batch_loss += loss
            
            # Проверка правильности предсказания
            predicted = np.argmax(outputs)
            if predicted == y:
                correct += 1
            
            # Обратное распространение
            gradients = self.compute_gradients(outputs, y)
            
            # Распространение ошибки назад через все слои
            current_gradients = gradients
            for layer in reversed(self.layers):
                for neuron in layer.neurons:
                    neuron.grad_weights = [0.0] * neuron.num_inputs
                    neuron.grad_bias = 0.0
                current_gradients = layer.backward(current_gradients)
        
        # Обновление весов всех слоев
        for layer in self.layers:
            layer.update(learning_rate)
        
        # Средние значения по батчу
        avg_loss = batch_loss / len(X_batch)
        accuracy = correct / len(X_batch)
        
        return avg_loss, accuracy
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, learning_rate=0.01,
              learning_rate_decay=0.95, early_stopping_patience=10, 
              stop_val_loss=0.5):
        """Обучение сети с валидацией"""
        
        print(f"Начинаем обучение...")
        print(f"Эпох: {epochs}, Размер батча: {batch_size}, LR: {learning_rate}")
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Случайное перемешивание данных
            indices = list(range(len(X_train)))
            random.shuffle(indices)
            X_train_shuffled = [X_train[i] for i in indices]
            y_train_shuffled = [y_train[i] for i in indices]
            
            train_loss = 0.0
            train_correct = 0
            num_batches = 0
            
            # Обучение по батчам
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                loss, accuracy = self.train_on_batch(X_batch, y_batch, learning_rate)
                
                train_loss += loss
                train_correct += accuracy * len(X_batch)
                num_batches += 1
            
            # Средние значения по эпохе
            avg_train_loss = train_loss / num_batches
            avg_train_accuracy = train_correct / len(X_train)
            
            # Валидация
            val_loss, val_accuracy = self.evaluate(X_val, y_val)
            
            # Сохранение истории
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_accuracy'].append(avg_train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Вывод прогресса
            print(f"Эпоха {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {avg_train_accuracy:.2%} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.2%}")
            
            # Ранняя остановка
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Ранняя остановка на эпохе {epoch+1}")
                    break

            if val_loss <= stop_val_loss:
                print(f"Граница ошибки достигнута на эпохе {epoch+1}")
                break
            
            # Уменьшение скорости обучения
            learning_rate *= learning_rate_decay
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Оценка точности на тестовых данных"""
        total_loss = 0.0
        correct = 0
        
        for X, y in zip(X_test, y_test):
            outputs = self.forward(X)
            loss = self.compute_loss(outputs, y)
            total_loss += loss
            
            predicted = np.argmax(outputs)
            if predicted == y:
                correct += 1
        
        avg_loss = total_loss / len(X_test)
        accuracy = correct / len(X_test)
        
        return avg_loss, accuracy

def create_network(input_size, num_classes):
    """Создание нейронной сети для классификации"""
    network = NeuralNetwork()
    
    # Первый скрытый слой: 128 нейронов с ReLU
    network.add_layer(DenseLayer(16, input_size, activation='relu'))
    
    # Второй скрытый слой: 64 нейрона с ReLU
    # network.add_layer(DenseLayer(16, 16, activation='relu'))
    # network.add_layer(DenseLayer(16, 16, activation='relu'))
    # network.add_layer(DenseLayer(16, 16, activation='relu'))
    
    # Выходной слой: num_classes нейронов с softmax (реализуем отдельно)
    # Для простоты используем линейный слой, а softmax в функции потерь
    network.add_layer(DenseLayer(num_classes, 16, activation='linear'))
    # network.add_layer(DenseLayer(num_classes, input_size, activation='linear'))
    
    return network

def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.set_title('Функция потерь во время обучения')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.set_title('Точность во время обучения')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("График обучения сохранен в 'training_history.png'")
    plt.show()

def load_battery_dataset(
    features_path: str="smartphone_battery_features.csv",
    targets_path: str="smartphone_battery_targets.csv"
) -> tuple[list[list[float]], list[int], dict]:
    """
    Загрузка датасета рекомендаций по батарее.
    """
    df_X = pd.read_csv(features_path)
    df_y = pd.read_csv(targets_path)

    X = df_X.select_dtypes(exclude="object")
    X = (X - X.mean()) / (X.std() + 1e-8)
    X = X.values.tolist()

    labels = df_y["recommended_action"]
    unique = sorted(labels.unique())
    mapping = {label: i for i, label in enumerate(unique)}
    y = [mapping[l] for l in labels]

    return X, y, mapping


if __name__ == "__main__":
    os.environ["TCL_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
    os.environ["TK_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

    random.seed(42)
    np.random.seed(42)

    # Загрузка разметки
    X, y, label_mapping = load_battery_dataset()

    # Создаем mapping меток в числовые индексы
    num_classes = len(label_mapping)
    
    print(f"Найдено классов: {num_classes}")
    print("Соответствие меток:", label_mapping)
    
    print(f"\nЗагружено {len(X)} примеров")
    print(f"Размер вектора признаков: {len(X[0])}")
    
    # Разделение на обучающую, валидационную и тестовую выборки
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {len(X_train)}")
    print(f"  Валидационная выборка: {len(X_val)}")
    print(f"  Тестовая выборка: {len(X_test)}")
    
    # Создаем нейронную сеть
    input_size = len(X[0])
    print(f"\nСоздаем нейронную сеть:")
    print(f"  Входной размер: {input_size}")
    print(f"  Количество классов: {num_classes}")
    
    nn = create_network(input_size, num_classes)
    
    # Обучение сети
    print("\n" + "="*70)
    print("НАЧИНАЕМ ОБУЧЕНИЕ С ГРАДИЕНТНЫМ СПУСКОМ")
    print("="*70)
    
    history = nn.train(
        X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=64,
        learning_rate=0.01,
        learning_rate_decay=0.98,
        early_stopping_patience=105,
        stop_val_loss=0.05
    )
    print("\nОбучение завершено!")
    
    # Оценка на тестовых данных
    test_loss, test_accuracy = nn.evaluate(X_test, y_test)
    print(f"\n" + "="*70)
    print(f"РЕЗУЛЬТАТЫ НА ТЕСТОВЫХ ДАННЫХ:")
    print(f"  Потери: {test_loss:.4f}")
    print(f"  Точность: {test_accuracy:.2%}")
    print("="*70)
    
    # Визуализация процесса обучения
    plot_training_history(history)
    
    # Примеры предсказаний
    print("\nПримеры предсказаний на тестовых данных:")
    print("-" * 50)
    
    for i in range(min(5, len(X_test))):
        predicted_idx, outputs = nn.predict(X_test[i])
        true_label_idx = y_test[i]
        
        # Находим исходные метки
        true_label = [k for k, v in label_mapping.items() if v == true_label_idx][0]
        predicted_label = [k for k, v in label_mapping.items() if v == predicted_idx][0]
        
        is_correct = "✓" if predicted_idx == true_label_idx else "✗"
        
        print(f"Пример {i+1}: {is_correct}")
        print(f"  Истинная метка: '{true_label}' (класс {true_label_idx})")
        print(f"  Предсказанная:  '{predicted_label}' (класс {predicted_idx})")
        
        # Вероятности по классам
        print(f"  Вероятности: ", end="")
        for j, prob in enumerate(outputs):
            label_name = [k for k, v in label_mapping.items() if v == j][0]
            print(f"{label_name}: {prob:.3f}  ", end="")
        print("\n")
