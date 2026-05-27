package com.lab6;

import java.util.Arrays;

import static com.lab6.ActivationFunction.softmax;

public class LogisticRegression {
    public int nIn;
    public int nOut;
    public double[][] W;
    public double[] b;

    public LogisticRegression(int nIn, int nOut) {
        this.nIn = nIn;
        this.nOut = nOut;
        W = new double[nOut][nIn];
        b = new double[nOut];
    }

    public double[][] train(double[][] X, int T[][], int miniBatchSize, double
            learningRate) {
        double[][] grad_W = new double[nOut][nIn];
        double[] grad_b = new double[nOut];
        double[][] dY = new double[miniBatchSize][nOut];
// train with SGD
// 1. calculate gradient of W, b
        for (int n = 0; n < miniBatchSize; n++) {
            double[] predicted_Y_ = output(X[n]);
            System.out.println("Y=" + Arrays.toString(predicted_Y_));
            for (int j = 0; j < nOut; j++) {
                dY[n][j] = predicted_Y_[j] - T[n][j];
                for (int i = 0; i < nIn; i++) {
                    grad_W[j][i] += dY[n][j] * X[n][i];
                }
                grad_b[j] += dY[n][j];
            }
        }
// 2. update params
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) {
                W[j][i] -= learningRate * grad_W[j][i] / miniBatchSize;
            }
            b[j] -= learningRate * grad_b[j] / miniBatchSize;
        }
        return dY;
    }

    public Integer[] predict(double[] x) {
        double[] y = output(x); // activate input data through learned networks
        Integer[] t = new Integer[nOut]; // output is the probability, so cast it to label
        int argmax = -1;
        double max = 0.;
        for (int i = 0; i < nOut; i++) {
            if (max < y[i]) {
                max = y[i];
                argmax = i;
            }
        }
        for (int i = 0; i < nOut; i++) {
            if (i == argmax) {
                t[i] = 1;
            } else {
                t[i] = 0;
            }
        }
        return t;
    }

    private double[] output(double[] x) {
        double[] preActivation = new double[nOut];
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) {
                preActivation[j] += W[j][i] * x[i];
            }
            preActivation[j] += b[j]; // linear output
        }
        return softmax(preActivation, nOut);
    }

    @Override
    public String toString() {
        return "b=" + Arrays.toString(b) + "; w=" +
                Arrays.deepToString(W);
    }
}