package com.lab9;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;

import static com.lab9.ActivationFunction.dsigmoid;
import static com.lab9.ActivationFunction.dtanh;
import static com.lab9.ActivationFunction.sigmoid;
import static com.lab9.ActivationFunction.tanh;
import static com.lab9.RandomGenerator.binomial;
import static com.lab9.RandomGenerator.uniform;

public class HiddenLayer {
    public int nIn;
    public int nOut;
    public double[][] W;
    public double[] b;
    public Random rng;
    public DoubleFunction<Double> activation;
    public DoubleFunction<Double> dactivation;

    public HiddenLayer(int nIn, int nOut, double[][] W, double[] b, Random
            rng, String activation) {
        if (rng == null) rng = new Random(1234); // seed random
        if (W == null) {
            W = new double[nOut][nIn];
            double w_ = 1. / nIn;
            for (int j = 0; j < nOut; j++) {
                for (int i = 0; i < nIn; i++) {
                    W[j][i] = uniform(-w_, w_, rng); // initialize W with uniform distribution
                }
            }
        }
        if (b == null) b = new double[nOut];
        this.nIn = nIn;
        this.nOut = nOut;
        this.W = W;
        this.b = b;
        this.rng = rng;
        if (activation == "sigmoid" || activation == null) {
            this.activation = (double x) -> sigmoid(x);
            this.dactivation = (double x) -> dsigmoid(x);
        } else if (activation == "tanh") {
            this.activation = (double x) -> tanh(x);
            this.dactivation = (double x) -> dtanh(x);
        } else {
            throw new IllegalArgumentException("activation function not supported");
        }
    }

    public double[] output(double[] x) {
        double[] y = new double[nOut];
        for (int j = 0; j < nOut; j++) {
            double preActivation = 0.;
            for (int i = 0; i < nIn; i++) {
                preActivation += W[j][i] * x[i];
            }
            preActivation += b[j];
            y[j] = activation.apply(preActivation);
        }
        return y;
    }

    public double[] forward(double[] x) {
        return output(x);
    }

    public double[][] backward(double[][] X, double[][] Z, double[][] dY,
                               double[][] Wprev, int miniBatchSize, double learningRate) {
        double[][] dZ = new double[miniBatchSize][nOut]; //backpropagation error
        double[][] grad_W = new double[nOut][nIn];
        double[] grad_b = new double[nOut];
// train with SGD
// calculate backpropagation error to get gradient of W, b
        for (int n = 0; n < miniBatchSize; n++) {
            for (int j = 0; j < nOut; j++) {
                for (int k = 0; k < dY[0].length; k++) { // k < ( nOut of previous layer )
                    dZ[n][j] += Wprev[k][j] * dY[n][k];
                }
                dZ[n][j] *= dactivation.apply(Z[n][j]);
                for (int i = 0; i < nIn; i++) {
                    grad_W[j][i] += dZ[n][j] * X[n][i];
                }
                grad_b[j] += dZ[n][j];
            }
        }
// update params
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) {
                W[j][i] -= learningRate * grad_W[j][i] / miniBatchSize;
            }
            b[j] -= learningRate * grad_b[j] / miniBatchSize;
        }
        return dZ;
    }

    @Override
    public String toString() {
        return "b=" + Arrays.toString(b) + "; w=" +
                Arrays.deepToString(W);
    }

    public int[] outputBinomial(int[] x, Random rng) {
        int[] y = new int[nOut];
        double[] xCast = new double[x.length];
        for (int i = 0; i < xCast.length; i++) {
            xCast[i] = (double) x[i];
        }
        double[] out = output(xCast);
        for (int j = 0; j < nOut; j++) {
            y[j] = binomial(1, out[j], rng);
        }
        return y;
    }
}