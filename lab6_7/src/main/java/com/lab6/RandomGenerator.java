package com.lab6;

import java.util.Random;

public class RandomGenerator {
    public static double uniform(double min, double max, Random rng) {
        return rng.nextDouble() * (max - min) + min;
    }
}