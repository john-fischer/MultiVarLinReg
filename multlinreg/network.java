package multlinreg;

import java.util.*;

public class network {

    Random rand = new Random();
    double[] m;
    double b;
    double lr;

    public network(int n) {
        m = new double[n];
        for (int i = 0; i < n; ++i) {
            m[i] = rand.nextDouble() % 10 * 100;
        }
        b = rand.nextDouble() % 10 * 10;
        lr = .00001;
    }

// a is the set of inputs
// bias is the bias
    public double sigmoid(double n)
    {return 1/(1+ Math.exp(n));}
    
    
    public double hypothesis(double[] a) {
        double result = 0;
        for (int i = 0; i < m.length; ++i) {
            result += a[i] * m[i];
        }
        return result + b;
    }

    public double hypothesis2(double[] a, double[] b, double bias) {
        double result = 0;
        for (int i = 0; i < a.length; ++i) {
            result += a[i] * b[i];
        }
        return result + bias;
    }

// this train function determines the amount of iterations we need
    public void train(int iter, double[][] x, double[] y) {
        for (int i = 0; i < iter; ++i) {
            train(x, y);
            //System.out.println((hypothesis(x[0]) - y[0]) * lr);
        }
    }

// this train function
    public void train(double[][] x, double[] y) {
        for (int i = 0; i < y.length; ++i) {
            train(x[i], y[i]);
        }
    }

// 
    public void train(double[] x, double y) {

        double error = (hypothesis(x) - y);
        //System.out.println(error);
        for (int i = 0; i < m.length; ++i) {
            m[i] -= (x[i] * error) * lr;
        }
        b -= error * lr;
    }
}
