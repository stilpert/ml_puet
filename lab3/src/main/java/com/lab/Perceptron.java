package com.lab;

import java.util.Arrays;

public class Perceptron {
    private int n;          // Кількість входів
    private double[] w;     // Вагові коефіцієнти
    private double b;       // Зміщення (bias), еквівалент порогу -w0

    // Конструктор
    public Perceptron(int n) {
        this.n = n;
        this.w = new double[n];
        this.b = 0; // Початкові вагові коефіцієнти та поріг беруться нульовими
    }

    // Метод, що проводить одну ітерацію навчання на одному екземплярі даних
    private int trainOneExample(double[] x, int target) {
        int classified = 0;
        double predict = b;

        // Обчислення лінійної комбінації (зваженої суми)
        for (int i = 0; i < n; i++) {
            predict += w[i] * x[i];
        }

        // Перевірка: чи збігається передбачення з цільовим значенням
        if (ActivationFunction.step(predict) == target) {
            classified = 1; // Класифіковано правильно
        } else {
            // Корекція ваг за алгоритмом навчання персептрона
            if (target == 1) {
                for (int i = 0; i < n; i++) {
                    w[i] += x[i];
                }
                b += 1;
            } else {
                for (int i = 0; i < n; i++) {
                    w[i] -= x[i];
                }
                b -= 1;
            }
        }
        return classified;
    }

    // Метод для навчання персептрона протягом кількох епох
    public void train(double[][] trainX, int[] trainT, int maxEpochs) {
        int epoch = 0;
        while (true) {
            int classifiedCount = 0;
            System.out.println("Epoch: " + epoch);

            // Прохід по всіх навчальних прикладах поточного циклу (епохи)
            for (int i = 0; i < trainX.length; i++) {
                classifiedCount += trainOneExample(trainX[i], trainT[i]);
            }

            // Виведення поточного стану ваг
            System.out.println(this);

            // Якщо всі приклади класифіковані правильно, зупиняємо навчання
            if (classifiedCount == trainX.length) {
                System.out.println("Навчання успішно завершено! Всі приклади класифіковано правильно.");
                break;
            }

            epoch++;
            if (epoch > maxEpochs) {
                System.out.println("Досягнуто ліміту епох навчання.");
                break;
            }
        }
    }

    // Метод, що обраховує результат після навчання для нових даних
    public int predict(double[] x) {
        double preActivation = b;
        for (int i = 0; i < n; i++) {
            preActivation += w[i] * x[i];
        }
        return ActivationFunction.step(preActivation);
    }

    // Форматоване виведення об'єкта
    @Override
    public String toString() {
        return "b = " + b + "; w = " + Arrays.toString(w);
    }
}