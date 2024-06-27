import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MLP_cls:
    def __init__(self, layer_sizes, activation_functions):
        # Инициализация сети с указанием размеров слоёв и функций активации
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        # Инициализация весов случайными значениями
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        # Инициализация смещений случайными значениями
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def sigmoid(self, z):
        # Сигмоида
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        # Производная сигмоиды
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        # Гиперболический тангенс
        return np.tanh(z)

    def tanh_derivative(self, z):
        # Производная гиперболического тангенса
        return 1 - np.tanh(z) ** 2

    def relu(self, z):
        # ReLU (Rectified Linear Unit)
        return np.maximum(0, z)

    def relu_derivative(self, z):
        # Производная ReLU
        return (z > 0) * 1

    def forward_propagation(self, input_data):
        # Прямое распространение
        for b, w, activation in zip(self.biases, self.weights, self.activation_functions):
            # Выбор функции активации и вычисление активации для текущего слоя
            if activation == 'sigmoid':
                input_data = self.sigmoid(np.dot(w, input_data) + b)
            elif activation == 'tanh':
                input_data = self.tanh(np.dot(w, input_data) + b)
            elif activation == 'relu':
                input_data = self.relu(np.dot(w, input_data) + b)
        return input_data

    def backpropagation(self, input_data, target):
        # Инициализация градиентов для весов и смещений нулями
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # Прямое распространение с сохранением активаций и взвешенных сумм (z)
        activation = input_data
        activations = [input_data]
        zs = []
        for b, w, activation_function in zip(self.biases, self.weights, self.activation_functions):
            z = np.dot(w, activation) + b
            zs.append(z)
            if activation_function == 'sigmoid':
                activation = self.sigmoid(z)
            elif activation_function == 'tanh':
                activation = self.tanh(z)
            elif activation_function == 'relu':
                activation = self.relu(z)
            activations.append(activation)

        # Обратное распространение
        # Вычисление ошибки на выходном слое
        delta = self.cost_derivative(activations[-1], target) * self.get_activation_derivative(activations[-1], self.activation_functions[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)

        # Вычисление ошибки на скрытых слоях
        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            activation_derivative = self.get_activation_derivative(z, self.activation_functions[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * activation_derivative
            gradient_b[-l] = delta
            gradient_w[-l] = np.dot(delta, activations[-l - 1].T)

        return gradient_b, gradient_w

    def get_activation_derivative(self, z, activation_function):
        # Получение производной функции активации
        if activation_function == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif activation_function == 'tanh':
            return self.tanh_derivative(z)
        elif activation_function == 'relu':
            return self.relu_derivative(z)

    def cost_derivative(self, output_activations, target):
        # Производная функции стоимости (ошибки)
        return output_activations - target

    def update_parameters(self, mini_batch, learning_rate):
        # Обновление параметров сети (весов и смещений) на основе градиентов
        sum_gradient_b = [np.zeros(b.shape) for b in self.biases]
        sum_gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
            sum_gradient_b = [nb + dnb for nb, dnb in zip(sum_gradient_b, delta_gradient_b)]
            sum_gradient_w = [nw + dnw for nw, dnw in zip(sum_gradient_w, delta_gradient_w)]

        # Обновление весов и смещений с использованием средних значений градиентов
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, sum_gradient_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, sum_gradient_b)]

class MLP_reg:
    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []
        self.activations = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(layer_sizes[i + 1]))
            self.activations.append(np.zeros(layer_sizes[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def derivative(self, x, activation_function):
        if activation_function == 'sigmoid':
            return x * (1 - x)
        elif activation_function == 'tanh':
            return 1 - np.power(x, 2)
        elif activation_function == 'relu':
            return (x > 0).astype(float)

    def forward(self, x):
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if self.activation_functions[i] == 'sigmoid':
                x = self.sigmoid(x)
            elif self.activation_functions[i] == 'tanh':
                x = self.tanh(x)
            elif self.activation_functions[i] == 'relu':
                x = self.relu(x)
            self.activations[i] = x
        return x

    def backward(self, x, y, learning_rate):
        output = self.forward(x)
        deltas = []
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                error = output - y
            else:
                error = np.dot(deltas[-1], self.weights[i + 1].T)
            delta = error * self.derivative(self.activations[i], self.activation_functions[i])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(x.T if i == 0 else self.activations[i - 1].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - ss_res / ss_tot
    
# Загрузка данных классификации
def load_data_cls():
    data_cls = pd.read_csv('csgo.csv')
    return data_cls 

data_cls = load_data_cls()

# Загрузка данных регрессии
def load_data_reg():
    data_reg = pd.read_csv('wines.csv')
    return data_reg

data_reg = load_data_reg()

# Заголовок приложения
st.title('MLP: Многослойный Персептрон')

st.title('Классификация')
# Отображение первых нескольких строк данных
st.subheader('Данные (классификация)')
st.write(data_cls.head())

# Выбор столбца для изменения
columns = data_cls.columns.tolist()
selected_column = st.selectbox('Выберите столбец для изменения (классификаиця)', columns)

# Ввод нового значения для всех строк выбранного столбца
new_value = st.text_input('Введите новое значение для выбранного столбца (классификация)')

# Кнопка для применения изменений
if st.button('Применить новое значение (классификация)'):
    data_cls[selected_column] = new_value
    st.write(f'Все значения в столбце {selected_column} изменены на {new_value}')
    st.write(data_cls.head())

# Раздел для обучения модели (пример)
st.subheader('Обучение модели (классификация)')
test_size = st.slider('Размер тестового набора (классификация )', 0.1, 0.5, 0.2)
random_state = st.number_input('Random state (классификация)', min_value=0, value=42)

if st.button('Обучить модель (классификация)'):
    X = data_cls.drop(['bomb_planted_True'], axis=1)
    y = data_cls['bomb_planted_True'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Подготовка данных для MLP
    X_train = [X_train[i].reshape(-1, 1) for i in range(X_train.shape[0])]
    y_train = [y_train[i].reshape(-1, 1) for i in range(y_train.shape[0])]
    training_data = list(zip(X_train, y_train))

    # Определение параметров сети
    input_size = X.shape[1]
    hidden_layers = [10, 5]
    output_size = 1

    # Создание и обучение MLP
    mlp = MLP_cls([input_size] + hidden_layers + [output_size], ['relu'] * len(hidden_layers) + ['sigmoid'])
    learning_rate = 0.01
    for epoch in range(20):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+10] for k in range(0, len(training_data), 10)]
        for mini_batch in mini_batches:
            mlp.update_parameters(mini_batch, learning_rate)

    def predict(mlp, X):
        predictions = [mlp.forward_propagation(x.reshape(-1, 1)) for x in X]
        return np.array(predictions).squeeze()

    predictions = predict(mlp, X_test)
    predictions_binary = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions_binary)

    st.write("Accuracy:", accuracy)
    print("Accuracy:", accuracy)

st.title('Регрессия')
# Отображение первых нескольких строк данных
st.subheader('Данные (регрессия)')
st.write(data_reg.head())

# Выбор столбца для изменения
columns2 = data_reg.columns.tolist()
selected_column2 = st.selectbox('Выберите столбец для изменения (регрессия)', columns2)

# Ввод нового значения для всех строк выбранного столбца
new_value2 = st.text_input('Введите новое значение для выбранного столбца (регрессия)')

# Кнопка для применения изменений
if st.button('Применить новое значение (регрессия)'):
    data_reg[selected_column2] = new_value2
    st.write(f'Все значения в столбце {selected_column2} изменены на {new_value2}')
    st.write(data_reg.head())

# Раздел для обучения модели (пример)
st.subheader('Обучение модели')
test_size2 = st.slider('Размер тестового набора (регрессия)', 0.1, 0.5, 0.2)
random_state2= st.number_input('Random state (регрессия)', min_value=0, value=42)

if st.button('Обучить модель (регрессия)'):
    def preprocess_data(trip_data):
        X = data_reg.drop(columns=['quality']).to_numpy()
        y = data_reg['quality'].to_numpy().reshape(-1, 1)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()
        return X, y

    # Подготовка данных
    X, y = preprocess_data(data_reg)

    # Инициализация и обучение модели
    layer_sizes = [X.shape[1], 10, 5, 1]  # Примерная архитектура сети
    activation_functions = ['tanh', 'relu', 'tanh']
    mlp = MLP_reg(layer_sizes, activation_functions)
    mlp.train(X, y, learning_rate=0.01, epochs=100)

    # Проверка модели
    y_pred = mlp.forward(X)
    r2 = mlp.r2_score(y, y_pred)
    st.write("R2:", r2)
    print(f"R2: {r2}")