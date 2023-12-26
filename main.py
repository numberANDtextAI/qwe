import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential([])
# model.add(tf.keras.layers.Flatten(input   _shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(x_train, y_train, epochs=50)
#
# model.save('handwritten.model')
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f"Точность на тестовых данных: {test_accuracy}")

model = tf.keras.models.load_model('handwritten.model')

# Предположим, у вас есть начальные данные x_train и y_train
# Определите их здесь или загрузите их из соответствующих источников

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

        feedback = input(f"Это цифра: {predicted_digit}. Это правильно? (да/нет): ")

        if feedback.lower() == 'нет':
            # Принятое предсказание считается неправильным, обновляем данные для переобучения модели
            correct_label = int(input("Введите правильную метку для этой цифры (0-9): "))
            y_train_updated = np.array([correct_label])
            x_train_updated = np.array([img.reshape(28, 28)])  # При необходимости, измените форму массива на нужную

            # Если x_train и y_train еще не определены, определите их здесь
            try:
                x_train = np.concatenate((x_train, x_train_updated))
                y_train = np.concatenate((y_train, y_train_updated))
            except NameError:
                x_train = x_train_updated
                y_train = y_train_updated

            # Переобучение модели
            model.fit(x_train, y_train, epochs=2)

            # Сохранение обновленной модели
            model.save('handwritten.model')

        elif feedback.lower() == 'да':
            # Принятое предсказание считается правильным, переходим к следующему изображению без изменения модели
            pass

    except Exception as e:
        print(f"Ошибка: {e}")

    finally:
        image_number += 1
