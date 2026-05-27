package com.lab6;

import java.util.Arrays;
import java.util.Random;

public class LetterRecognitionRunner {
    public static void main(String[] args) {
        final Random rng = new Random(42); // Фіксований seed для відтворюваності

        // 1. Параметри архітектури мережі
        final int nIn = 25;       // Розмірність вхідного вектору (сітка літери 5x5)
        final int nHidden = 8;    // Кількість нейронів у прихованому шарі (підбирається експериментально)
        final int patterns = 3;   // Кількість класів (Літери: 0 - 'А', 1 - 'Б', 2 - 'В')
        final int nOut = patterns;

        final int epochs = 800;
        final double learningRate = 0.15;
        final int miniBatchSize = 3; // Навчаємо батчем, що дорівнює розміру вибірки

        // 2. Навчальна вибірка (Матриці 5х5 розгорнуті в лінію)
        // 1.0 означає зафарбований піксель, 0.0 - пустий
        double[][] train_X = {
                // Літера "А"
                {
                        0.,1.,1.,1.,0.,
                        1.,0.,0.,0.,1.,
                        1.,1.,1.,1.,1.,
                        1.,0.,0.,0.,1.,
                        1.,0.,0.,0.,1.
                },
                // Літера "Б"
                {
                        1.,1.,1.,1.,1.,
                        1.,0.,0.,0.,0.,
                        1.,1.,1.,1.,0.,
                        1.,0.,0.,0.,1.,
                        1.,1.,1.,1.,0.
                },
                // Літера "В"
                {
                        1.,1.,1.,1.,0.,
                        1.,0.,0.,0.,1.,
                        1.,1.,1.,1.,0.,
                        1.,0.,0.,0.,1.,
                        1.,1.,1.,1.,0.
                }
        };

        // Цільові значення (One-Hot Encoding)
        // {1, 0, 0} -> Клас 0 ('А')
        // {0, 1, 0} -> Клас 1 ('Б')
        // {0, 0, 1} -> Клас 2 ('В')
        int[][] train_T = {
                {1, 0, 0}, // А
                {0, 1, 0}, // Б
                {0, 0, 1}  // В
        };

        // 3. Тестова вибірка (Літери з невеликим «шумом» або викривленням)
        double[][] test_X = {
                // Викривлена "А" (пропущено один піксель збоку)
                {
                        0.,1.,1.,1.,0.,
                        1.,0.,0.,0.,0., // тут 0 замість 1
                        1.,1.,1.,1.,1.,
                        1.,0.,0.,0.,1.,
                        1.,0.,0.,0.,1.
                },
                // Викривлена "Б" (верхня лінія коротша)
                {
                        1.,1.,1.,0.,0., // тут 0 замість 1
                        1.,0.,0.,0.,0.,
                        1.,1.,1.,1.,0.,
                        1.,0.,0.,0.,1.,
                        1.,1.,1.,1.,0.
                }
        };

        int[][] test_T = {
                {1, 0, 0}, // Очікується А
                {0, 1, 0}  // Очікується Б
        };

        // 4. Ініціалізація та навчання моделі МЛП
        System.out.println("--- Ініціалізація MLP для розпізнавання літер ---");
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(nIn, nHidden, nOut, rng);

        System.out.println("--- Початок навчання ---");
        mlp.train(train_X, train_T, epochs, miniBatchSize, learningRate);

        // 5. Тестування розпізнавання викривлених образів
        System.out.println("\n=== ТЕСТУВАННЯ НА ВИКРИВЛЕНИХ ОБРАЗАХ ===");
        String[] letterLabels = {"А", "Б", "В"};

        for (int i = 0; i < test_X.length; i++) {
            Integer[] prediction = mlp.predict(test_X[i]);

            int predictedClass = -1;
            for (int c = 0; c < prediction.length; c++) {
                if (prediction[c] == 1) {
                    predictedClass = c;
                    break;
                }
            }

            int actualClass = -1;
            for (int c = 0; c < test_T[i].length; c++) {
                if (test_T[i][c] == 1) {
                    actualClass = c;
                    break;
                }
            }

            System.out.println("Тестовий зразок №" + (i + 1));
            System.out.println("  Очікувано: " + letterLabels[actualClass]);
            System.out.println("  Розпізнано: " + (predictedClass != -1 ? letterLabels[predictedClass] : "Невідома літера"));
            System.out.println("  Вихідний вектор: " + Arrays.toString(prediction));
        }
    }
}