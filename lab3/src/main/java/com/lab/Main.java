package com.lab;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        final int maxEpochs = 20; // Для простих функцій цього цілком достатньо
        int n = 2;                // Два входи (x1, x2)

        // Загальна матриця входів для всіх булевих функцій двох змінних
        double[][] trainX = {
                {0, 0}, // x1=0, x2=0
                {0, 1}, // x1=0, x2=1
                {1, 0}, // x1=1, x2=0
                {1, 1}  // x1=1, x2=1
        };

        // 1. Масиви цільових значень (Target) для кожної з 5 функцій Варіанта №5
        int[] targetF5 = {1, 0, 1, 0}; // F2,5 = NOT x2 (Заперечення другого операнда)
        int[] targetF0 = {0, 0, 0, 0}; // F2,0 = 0      (Тотожний нуль)
        int[] targetF6 = {0, 1, 1, 0}; // F2,6 = XOR    (Виключне АБО)
        int[] targetF8 = {0, 0, 0, 1}; // F2,8 = AND    (Кон'юнкція)
        int[] targetF10 = {0, 1, 0, 1}; // F2,10 = x2    (Повторення другого операнда)

        // Назви функцій для гарного виведення в консоль
        String[] functionNames = {
                "F2,5 = NOT x2 (Заперечення другого операнда)",
                "F2,0 = 0 (Тотожний нуль)",
                "F2,6 = XOR (Виключне АБО) *Лінійно нероздільна*",
                "F2,8 = AND (Кон'юнкція)",
                "F2,10 = x2 (Повторення другого операнда)"
        };

        int[][] allTargets = {targetF5, targetF0, targetF6, targetF8, targetF10};

        // 2. Цикл перебору та навчання персептрона для кожної функції
        for (int f = 0; f < allTargets.length; f++) {
            System.out.println("\n=======================================================");
            System.out.println("НАВЧАННЯ ДЛЯ ФУНКЦІЇ: " + functionNames[f]);
            System.out.println("=======================================================");

            // Створюємо новий чистий персептрон для нової функції
            Perceptron classifier = new Perceptron(n);
            int[] currentTarget = allTargets[f];

            // Запускаємо процес навчання
            classifier.train(trainX, currentTarget, maxEpochs);

            // Перевірка результатів
            System.out.println("\n--- Перевірка результатів (Predict) ---");
            for (int i = 0; i < trainX.length; i++) {
                int output = classifier.predict(trainX[i]);
                System.out.println("Вхід: " + Arrays.toString(trainX[i]) +
                        " -> Отримано: " + output +
                        " (Очікувалось: " + currentTarget[i] + ")");
            }
        }
    }
}