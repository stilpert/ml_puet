package com.lab6;

import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        final Random rng = new Random(123); // seed random
//
// Declare variables and constants
//
        final int patterns = 2;
        final int test_N = 4;
        final int nIn = 2;
        final int nOut = patterns;
        double[][] train_X;
        int[][] train_T;
        double[][] test_X;
        Integer[][] test_T;
        Integer[][] predicted_T = new Integer[test_N][nOut];
//        final int nHidden = 3;
//        final int epochs = 500;
//        double learningRate = 0.1;
//        final int miniBatchSize = 2;
        final int nHidden = 5;       // Збільшуємо кількість нейронів прихованого шару
        final int epochs = 1000;     // Збільшуємо кількість епох для кращої збіжності
        double learningRate = 0.05;  // Зменшуємо крок навчання для плавнішого оновлення ваг
        final int miniBatchSize = 4; // Збільшуємо розмір батчу до розміру всієї вибірки (Full Batch)
        train_X = new double[][]{
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        train_T = new int[][]{
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
        };
        test_X = new double[][]{
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        test_T = new Integer[][]{
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
        };
//
// Build Multi-Layer Perceptron model
//
// construct
        MultiLayerPerceptron classifier = new MultiLayerPerceptron(nIn,
                nHidden, nOut, rng);
//train
        classifier.train(train_X, train_T, epochs, miniBatchSize, learningRate);
// test
        System.out.println("===TEST===");
        for (int i = 0; i < test_N; i++) {
            System.out.println(Arrays.toString(test_X[i]));
            predicted_T[i] = classifier.predict(test_X[i]);
            System.out.println(Arrays.toString(predicted_T[i]));
        }
//
// Evaluate the model
//
        int[][] confusionMatrix = new int[patterns][patterns];
        double accuracy = 0.;
        double[] precision = new double[patterns];
        double[] recall = new double[patterns];
        for (int i = 0; i < test_N; i++) {
            int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
            int actual_ = Arrays.asList(test_T[i]).indexOf(1);
            confusionMatrix[actual_][predicted_] += 1;
        }
        for (int i = 0; i < patterns; i++) {
            double col_ = 0.;
            double row_ = 0.;
            for (int j = 0; j < patterns; j++) {
                if (i == j) {
                    accuracy += confusionMatrix[i][j];
                    precision[i] += confusionMatrix[j][i];
                    recall[i] += confusionMatrix[i][j];
                }
                col_ += confusionMatrix[j][i];
                row_ += confusionMatrix[i][j];
            }
            precision[i] /= col_;
            recall[i] /= row_;
        }
        accuracy /= test_N;
        System.out.println("--------------------");
        System.out.println("MLP model evaluation");
        System.out.println("--------------------");
        System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) {
            System.out.printf(" class %d: %.1f %%\n", i + 1, precision[i] * 100);
        }
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) {
            System.out.printf(" class %d: %.1f %%\n", i + 1, recall[i] * 100);
        }
    }
}
