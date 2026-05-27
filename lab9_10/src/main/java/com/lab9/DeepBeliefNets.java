package com.lab9;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DeepBeliefNets {
    public int nIn;
    public int[] hiddenLayerSizes;
    public int nOut;
    public int nLayers;
    public RestrictedBoltzmannMachines[] rbmLayers;
    public HiddenLayer[] sigmoidLayers;
    public LogisticRegression logisticLayer;
    public Random rng;

    public DeepBeliefNets(int nIn, int[] hiddenLayerSizes, int nOut,
                          Random rng) {
        if (rng == null) rng = new Random(1234);
        this.nIn = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.nOut = nOut;
        this.nLayers = hiddenLayerSizes.length;
        this.sigmoidLayers = new HiddenLayer[nLayers];
        this.rbmLayers = new RestrictedBoltzmannMachines[nLayers];
        this.rng = rng;
// construct multi-layer
        for (int i = 0; i < nLayers; i++) {
            int nIn_;
            if (i == 0) nIn_ = nIn;
            else nIn_ = hiddenLayerSizes[i - 1];
// construct hidden layers with sigmoid function
// weight matrices and bias vectors will be shared with RBM layers
            sigmoidLayers[i] = new HiddenLayer(nIn_, hiddenLayerSizes[i],
                    null, null, rng, "sigmoid");
// construct RBM layers
            rbmLayers[i] = new RestrictedBoltzmannMachines(nIn_,
                    hiddenLayerSizes[i], sigmoidLayers[i].W, sigmoidLayers[i].b, null, rng);
        }
// logistic regression layer for output
        logisticLayer = new LogisticRegression(hiddenLayerSizes[nLayers -
                1], nOut);
    }

    public void pretrain(int[][][] X, int minibatchSize, int minibatch_N, int
            epochs, double learningRate, int k) {
        for (int layer = 0; layer < nLayers; layer++) { // pre-train layer-wise
            System.out.println("[Pretrain] Starting layer " + layer);
            for (int epoch = 0; epoch < epochs; epoch++) {
                double epochError = 0.0;
                for (int batch = 0; batch < minibatch_N; batch++) {
                    int[][] X_ = new int[minibatchSize][nIn];
                    int[][] prevLayerX_;
// Set input data for current layer
                    if (layer == 0) {
                        X_ = X[batch];
                    } else {
                        prevLayerX_ = X_;
                        X_ = new int[minibatchSize][hiddenLayerSizes[layer - 1]];
                        for (int i = 0; i < minibatchSize; i++) {
                            X_[i] = sigmoidLayers[layer - 1].outputBinomial(prevLayerX_[i], rng);
                        }
                    }
                    rbmLayers[layer].contrastiveDivergence(X_, minibatchSize, learningRate, k);
                    double reconError = rbmLayers[layer].getReconstructionError(X_);
                    epochError += reconError;
                    System.out.printf("[Pretrain] Layer %d, Epoch %d, Batch %d, Reconstruction Error: %.6f\n", layer, epoch, batch, reconError);
                }
                System.out.printf("[Pretrain] Layer %d, Epoch %d, Avg Reconstruction Error: %.6f\n", layer, epoch, epochError / minibatch_N);
            }
        }
    }

    public void finetune(double[][] X, int[][] T, int minibatchSize, double
            learningRate) {
        List<double[][]> layerInputs = new ArrayList<>(nLayers + 1);
        layerInputs.add(X);
        double[][] Z = new double[0][0];
        double[][] dY;
// forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_; // layer input
            double[][] Z_ = new double[minibatchSize][hiddenLayerSizes[layer]];
            for (int n = 0; n < minibatchSize; n++) {
                if (layer == 0) {
                    x_ = X[n];
                } else {
                    x_ = Z[n];
                }
                Z_[n] = sigmoidLayers[layer].forward(x_);
            }
            Z = Z_;
            layerInputs.add(Z.clone());
        }
// forward & backward output layer
        dY = logisticLayer.train(Z, T, minibatchSize, learningRate);
        // Logging for finetune: print average loss if available
        double avgLoss = 0.0;
        if (dY != null && dY.length > 0 && dY[0].length > 0) {
            for (int i = 0; i < dY.length; i++) {
                for (int j = 0; j < dY[i].length; j++) {
                    avgLoss += Math.abs(dY[i][j]);
                }
            }
            avgLoss /= (dY.length * dY[0].length);
            System.out.printf("[Finetune] Avg output layer delta (proxy for loss): %.6f\n", avgLoss);
        } else {
            System.out.println("[Finetune] Completed output layer training for minibatch.");
        }
// backward hidden layers
        double[][] Wprev;
        double[][] dZ = new double[0][0];
        for (int layer = nLayers - 1; layer >= 0; layer--) {
            if (layer == nLayers - 1) {
                Wprev = logisticLayer.W;
            } else {
                Wprev = sigmoidLayers[layer + 1].W;
                dY = dZ.clone();
            }
            dZ = sigmoidLayers[layer].backward(layerInputs.get(layer),
                    layerInputs.get(layer + 1), dY, Wprev, minibatchSize, learningRate);
        }
        System.out.println("[Finetune] Completed backward pass for minibatch.");
    }

    public Integer[] predict(double[] x) {
        double[] z = new double[0];
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_;
            if (layer == 0) {
                x_ = x;
            } else {
                x_ = z.clone();
            }
            z = sigmoidLayers[layer].forward(x_);
        }
        return logisticLayer.predict(z);
    }
}