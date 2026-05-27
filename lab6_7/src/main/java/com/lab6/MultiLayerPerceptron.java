package com.lab6;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultiLayerPerceptron {
    public int nIn;
    public int nHidden;
    public int nOut;
    public HiddenLayer hiddenLayer;
    public LogisticRegression logisticLayer;
    public Random rng;

    public MultiLayerPerceptron(int nIn, int nHidden, int nOut, Random
            rng) {
        this.nIn = nIn;
        this.nHidden = nHidden;
        this.nOut = nOut;
        if (rng == null) rng = new Random(1234);
        this.rng = rng;
// construct hidden layer with tanh as activation function
        hiddenLayer = new HiddenLayer(nIn, nHidden, null, null, rng,
                "tanh"); // sigmoid or tanh
// construct output layer i.e. multi-class logistic layer
        logisticLayer = new LogisticRegression(nHidden, nOut);
    }

    public void train(double[][] trainX, int[][] trainT, int maxEpochs, int
            miniBatchSize, double learningRate) {
        int train_N = trainX.length;
        int miniBatchNumbers = train_N / miniBatchSize;
        double[][][] trainXminiBatch = new double[miniBatchNumbers]
                [miniBatchSize][nIn];
        int[][][] train_T_minibatch = new int[miniBatchNumbers]
                [miniBatchSize][nOut];
        List<Integer> miniBatchIndex = new ArrayList<Integer>();
        for (int i = 0; i < train_N; i++) {
            miniBatchIndex.add(i);
        }
        Collections.shuffle(miniBatchIndex, rng);
// create minibatches
        for (int i = 0; i < miniBatchNumbers; i++) {
            for (int j = 0; j < miniBatchSize; j++) {
                trainXminiBatch[i][j] = trainX[miniBatchIndex.get(i *
                        miniBatchSize + j)];
                train_T_minibatch[i][j] = trainT[miniBatchIndex.get(i *
                        miniBatchSize + j)];
            }
        }
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            System.out.println("Epoch " + epoch);
            for (int batch = 0; batch < miniBatchNumbers; batch++) {
                trainOneBatch(trainXminiBatch[batch],
                        train_T_minibatch[batch], miniBatchSize, learningRate);
            }
            System.out.println(this);
        }
    }

    public Integer[] predict(double[] x) {
        double[] z = hiddenLayer.output(x);
        return logisticLayer.predict(z);
    }

    private void trainOneBatch(double[][] X, int T[][], int miniBatchSize,
                               double learningRate) {
        System.out.println("X=" + Arrays.deepToString(X));
        double[][] Z = new double[miniBatchSize][nIn]; // outputs of hidden layer (= inputs of output layer)
        double[][] dY;
// forward hidden layer
        for (int n = 0; n < miniBatchSize; n++) {
            Z[n] = hiddenLayer.forward(X[n]); // activate input units
        }
// forward & backward output layer
        dY = logisticLayer.train(Z, T, miniBatchSize, learningRate);
// backward hidden layer (backpropagate)
        hiddenLayer.backward(X, Z, dY, logisticLayer.W, miniBatchSize,
                learningRate);
    }

    @Override
    public String toString() {
        return "HiddenLayer:\n" + hiddenLayer + "\n" +
                "LogisticLayer:\n" + logisticLayer;
    }
}