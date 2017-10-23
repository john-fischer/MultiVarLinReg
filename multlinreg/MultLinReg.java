package multlinreg;

import java.util.*;
import java.io.*;

public class MultLinReg implements Serializable {

    public static void main(String[] args) {
        int N = 5;
        network nn = new network(N);
        Random rand = new Random(); //rand.nextDouble() % 10 * 100;
        double bias = 3;//rand.nextDouble() % 10 * 100;
        double[] slopes = new double[N];
        for (int i = 0; i < N; ++i) {
            slopes[i] = rand.nextDouble() % 10 * 100;
        }
        double[][] input = new double[15][N];
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                input[i][j] = rand.nextDouble() % 10 * 100;
            }
        }
        double[] output = new double[N];
        for (int j = 0; j < N; ++j) {
            output[j] = nn.hypothesis2(input[j], slopes, bias);
        }
       
        System.out.println("expected output : " + nn.hypothesis2(slopes,input[0], nn.b));
        System.out.println("our output : " + (nn.hypothesis2(nn.m,input[0], nn.b)));
        System.out.println("the b pre-training : " + nn.b);
        System.out.println("slopes : " + Arrays.toString(slopes));
        System.out.println("the weights pre-training : " + Arrays.toString(nn.m));

        nn.train(1000000, input, output);
        System.out.println("expected output : " + nn.hypothesis2(slopes,input[0], nn.b));
        System.out.println("our output : " + (nn.hypothesis2(nn.m,input[0], nn.b)));
        System.out.println("bias : " + bias);
        System.out.println("the b post-training : " + nn.b);
        System.out.println("slopes : " + Arrays.toString(slopes));
        System.out.println("the weights post-training : " + Arrays.toString(nn.m));

    }
}