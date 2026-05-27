package com.lab8;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static com.lab8.RandomGenerator.binomial;

public class Main {
    public static void main(String[] args) {
        final Random rng = new Random(123);
//
// Declare variables and constants
//
        int trainNEach = 200; // for demo
        int testNEach = 2; // for demo
        int nVisibleEach = 4; // for demo
        double pNoiseTraining = 0.05; // for demo
        double pNoiseTest = 0.25; // for demo
        final int patterns = 3;
        final int trainN = trainNEach * patterns;
        final int testN = testNEach * patterns;
        final int nVisible = nVisibleEach * patterns;
        int nHidden = 12;
        int[][] train_X = new int[trainN][nVisible];
        int[][] test_X = new int[testN][nVisible];
        double[][] reconstructed_X = new double[testN][nVisible];
        int epochs = 1000;
        double learningRate = 0.2;
        int miniBatchSize = 10;
        final int miniBatchN = trainN / miniBatchSize;
        int[][][] trainXminiBatch = new int[miniBatchN][miniBatchSize][nVisible];
        List<Integer> miniBatchIndex = new ArrayList<>();
        for (int i = 0; i < trainN; i++) miniBatchIndex.add(i);
        Collections.shuffle(miniBatchIndex, rng);
//
// Create training data and test data for demo.
// Data without noise would be:
// class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
// class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
// class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
// and to each data, we add some noise.
// For example, one of the data in class 1 could be:
// [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
//
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < trainNEach; n++) {
                int n_ = pattern * trainNEach + n;
                for (int i = 0; i < nVisible; i++) {
                    if ((n_ >= trainNEach * pattern && n_ < trainNEach * (pattern +
                            1)) &&
                            (i >= nVisibleEach * pattern && i < nVisibleEach * (pattern
                                    + 1))) {
                        train_X[n_][i] = binomial(1, 1 - pNoiseTraining, rng);
                    } else {
                        train_X[n_][i] = binomial(1, pNoiseTraining, rng);
                    }
                }
            }
            for (int n = 0; n < testNEach; n++) {
                int n_ = pattern * testNEach + n;
                for (int i = 0; i < nVisible; i++) {
                    if ((n_ >= testNEach * pattern && n_ < testNEach * (pattern + 1)
                    ) &&
                            (i >= nVisibleEach * pattern && i < nVisibleEach * (pattern
                                    + 1))) {
                        test_X[n_][i] = binomial(1, 1 - pNoiseTest, rng);
                    } else {
                        test_X[n_][i] = binomial(1, pNoiseTest, rng);
                    }
                }
            }
        }
// create minibatches
        for (int i = 0; i < miniBatchN; i++) {
            for (int j = 0; j < miniBatchSize; j++) {
                trainXminiBatch[i][j] = train_X[miniBatchIndex.get(i * miniBatchSize
                        + j)];
            }
        }
//
// Build Restricted Boltzmann Machine Model
//
// construct RBM
        RestrictedBoltzmannMachines nn = new
                RestrictedBoltzmannMachines(nVisible, nHidden, null, null, null, rng);
// train with contrastive divergence
        System.out.println("--- Початок навчання RBM ---");
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < miniBatchN; batch++) {
                nn.contrastiveDivergence(trainXminiBatch[batch], miniBatchSize, learningRate, 1);
            }

            // Виводимо лог кожні 100 епох
            if (epoch == 0 || (epoch + 1) % 100 == 0) {
                double totalError = 0;
                // Порахуємо середню помилку по всіх батчах вибірки
                for (int batch = 0; batch < miniBatchN; batch++) {
                    totalError += nn.getReconstructionError(trainXminiBatch[batch]);
                }
                double meanError = totalError / miniBatchN;
                System.out.printf("Епоха %4d / %d | Середня помилка реконструкції (MSE): %.6f | LR: %.5f\n",
                        (epoch + 1), epochs, meanError, learningRate);
            }

            learningRate *= 0.995; // Зменшення кроку навчання [cite: 580]
        }
        System.out.println("--- Навчання завершено ---");
// test (reconstruct noised data)
        for (int i = 0; i < testN; i++) {
            reconstructed_X[i] = nn.reconstruct(test_X[i]);
        }
// evaluation
        System.out.println("-----------------------------------");
        System.out.println("RBM model reconstruction evaluation");
        System.out.println("-----------------------------------");
        for (int pattern = 0; pattern < patterns; pattern++) {
            System.out.printf("Class%d\n", pattern + 1);
            for (int n = 0; n < testNEach; n++) {
                int n_ = pattern * testNEach + n;
                System.out.print(Arrays.toString(test_X[n_]) + " -> ");
                System.out.print("[");
                for (int i = 0; i < nVisible - 1; i++) {
                    System.out.printf("%.5f, ", reconstructed_X[n_][i]);
                }
                System.out.printf("%.5f]\n", reconstructed_X[n_][nVisible - 1]);
            }
            System.out.println();
        }
    }
}
