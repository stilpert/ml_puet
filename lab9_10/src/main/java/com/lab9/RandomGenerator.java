package com.lab9;

import java.util.Random;

public class RandomGenerator {
    public static int binomial(int n, double p, Random rng) {
        if (p < 0 || p > 1) return 0;
        int c = 0;
        double r;
        for (int i = 0; i < n; i++) {
            r = rng.nextDouble();
            if (r < p) c++;
        }
        return c;
    }

    public static double uniform(double min, double max, Random rng) {
        return rng.nextDouble() * (max - min) + min;
    }
}