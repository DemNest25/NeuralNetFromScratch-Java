import java.util.*;

public class NeuralNetFromScratch {

    static class Math2 {
        static double[][] matmul(double[][] A, double[][] B) {
            int n = A.length, m = A[0].length, p = B[0].length;
            double[][] C = new double[n][p];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < m; k++) {
                    double aik = A[i][k];
                    for (int j = 0; j < p; j++) C[i][j] += aik * B[k][j];
                }
            }
            return C;
        }
        static double[][] addBiasRowwise(double[][] A, double[] b) {
            int n = A.length, m = A[0].length;
            double[][] C = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) C[i][j] = A[i][j] + b[j];
            return C;
        }
        static double[][] transpose(double[][] A) {
            int n = A.length, m = A[0].length;
            double[][] T = new double[m][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) T[j][i] = A[i][j];
            return T;
        }
        static double[][] hadamard(double[][] A, double[][] B) {
            int n = A.length, m = A[0].length;
            double[][] C = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) C[i][j] = A[i][j] * B[i][j];
            return C;
        }
        static double[][] scalar(double[][] A, double s) {
            int n = A.length, m = A[0].length;
            double[][] C = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) C[i][j] = A[i][j] * s;
            return C;
        }
        static double[][] add(double[][] A, double[][] B) {
            int n = A.length, m = A[0].length;
            double[][] C = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j];
            return C;
        }
        static double[] meanColumns(double[][] A) {
            int n = A.length, m = A[0].length;
            double[] mean = new double[m];
            for (int j = 0; j < m; j++) {
                double s = 0.0;
                for (int i = 0; i < n; i++) s += A[i][j];
                mean[j] = s / n;
            }
            return mean;
        }
        static double[][] copy(double[][] A) {
            int n = A.length, m = A[0].length; double[][] C = new double[n][m];
            for (int i = 0; i < n; i++) System.arraycopy(A[i], 0, C[i], 0, m);
            return C;
        }
        static double[][] zeros(int n, int m) {
            return new double[n][m];
        }
        static double[][] from2D(double[][] a){ return a; }
    }

    // ===== Interfaces base =====
    interface Layer {
        double[][] forward(double[][] X);
        double[][] backward(double[][] dOut);
        void step(double lr, double l2);
    }

    interface Activation {
        double[][] forward(double[][] Z);
        double[][] backward(double[][] dA);
    }

    // ===== Capa Densa =====
    static class Dense implements Layer {
        final int in, out;
        double[][] W; 
        double[] b;   
        double[][] Xcache;
        double[][] dW;
        double[] db;
        Random rnd = new Random(42);

        Dense(int in, int out) {
            this.in = in; this.out = out;
            W = new double[in][out];
            b = new double[out];

            // Inicialización He/Glorot simple
            double limit = Math.sqrt(6.0/(in + out));
            for (int i = 0; i < in; i++)
                for (int j = 0; j < out; j++) W[i][j] = (rnd.nextDouble()*2-1)*limit;
        }
        public double[][] forward(double[][] X) {
            Xcache = Math2.copy(X);
            double[][] Z = Math2.matmul(X, W);
            return Math2.addBiasRowwise(Z, b);
        }
        public double[][] backward(double[][] dZ) {
            int N = Xcache.length;
            // Gradientes
            double[][] XT = Math2.transpose(Xcache);
            dW = Math2.scalar(Math2.matmul(XT, dZ), 1.0 / N);
            db = Math2.meanColumns(dZ);
            // Gradiente a la entrada
            double[][] WT = Math2.transpose(W);
            return Math2.matmul(dZ, WT);
        }
        public void step(double lr, double l2) {
            
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    double reg = l2 * W[i][j];
                    W[i][j] -= lr * (dW[i][j] + reg);
                }
            }
            for (int j = 0; j < out; j++) b[j] -= lr * db[j];
        }
    }

    // ===== Activaciones =====
    static class ReLU implements Activation {
        double[][] mask; // 1 donde Z>0
        public double[][] forward(double[][] Z) {
            int n = Z.length, m = Z[0].length;
            mask = new double[n][m];
            double[][] A = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) {
                    if (Z[i][j] > 0) { A[i][j] = Z[i][j]; mask[i][j] = 1.0; }
                    else { A[i][j] = 0.0; }
                }
            return A;
        }
        public double[][] backward(double[][] dA) {
            return Math2.hadamard(dA, mask);
        }
    }

    static class Sigmoid implements Activation {
        double[][] A;
        public double[][] forward(double[][] Z) {
            int n = Z.length, m = Z[0].length;
            A = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) A[i][j] = 1.0 / (1.0 + Math.exp(-Z[i][j]));
            return A;
        }
        public double[][] backward(double[][] dA) {
            int n = A.length, m = A[0].length;
            double[][] dZ = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) dZ[i][j] = dA[i][j] * A[i][j] * (1 - A[i][j]);
            return dZ;
        }
    }

    // ===== Pérdida BCE (para binario). Devuelve dA =====
    static class BinaryCrossEntropy {
        double eps = 1e-12;
        double lastLoss;

        // yTrue y yPred son [N x 1] donde y∈{0,1}
        double loss(double[][] yTrue, double[][] yPred) {
            int N = yTrue.length;
            double s = 0.0;
            for (int i = 0; i < N; i++) {
                double y = yTrue[i][0];
                double p = Math.min(1-eps, Math.max(eps, yPred[i][0]));
                s += -( y * Math.log(p) + (1 - y) * Math.log(1 - p) );
            }
            lastLoss = s / N;
            return lastLoss;
        }

        // dL/dA (para usar con backward de Sigmoid)
        double[][] dA(double[][] yTrue, double[][] yPred) {
            int N = yTrue.length;
            double[][] dA = new double[N][1];
            for (int i = 0; i < N; i++) {
                double y = yTrue[i][0];
                double p = Math.min(1-eps, Math.max(eps, yPred[i][0]));
                dA[i][0] = (p - y) / (p * (1 - p)) / N; 
            }
            return dA;
        }
    }

    // ===== Optimizador SGD =====
    static class SGD {
        double lr, l2;
        SGD(double lr, double l2) { this.lr = lr; this.l2 = l2; }
        void step(List<Layer> layers) {
            for (Layer L : layers) L.step(lr, l2);
        }
    }

    // ===== Modelo secuencial =====
    static class Sequential {
        List<Layer> layers = new ArrayList<>();
        List<Activation> acts = new ArrayList<>();

        double[][] forward(double[][] X) {
            double[][] out = X;
            for (int i = 0; i < layers.size(); i++) {
                out = layers.get(i).forward(out);
                if (i < acts.size() && acts.get(i) != null) out = acts.get(i).forward(out);
            }
            return out;
        }
        void backward(double[][] dOut) {
            double[][] grad = dOut;
            for (int i = layers.size()-1; i >= 0; i--) {
                if (i < acts.size() && acts.get(i) != null) grad = acts.get(i).backward(grad);
                grad = layers.get(i).backward(grad);
            }
        }
    }

    // ===== Demostración con XOR =====
    public static void main(String[] args) {
        double[][] X = {
                {0,0}, {0,1}, {1,0}, {1,1}
        };
        double[][] y = {
                {0}, {1}, {1}, {0}
        };

        Sequential model = new Sequential();
        model.layers.add(new Dense(2, 4));
        model.acts.add(new ReLU());
        model.layers.add(new Dense(4, 1));
        model.acts.add(new Sigmoid());

        BinaryCrossEntropy loss = new BinaryCrossEntropy();
        SGD opt = new SGD(0.5, 1e-4); 

        int epochs = 10000;
        for (int e = 1; e <= epochs; e++) {
            double[][] yPred = model.forward(X);
            double L = loss.loss(y, yPred);
            double[][] dA = loss.dA(y, yPred);
            model.backward(dA);
            opt.step(model.layers);

            if (e % 1000 == 0) {
                System.out.printf("Epoch %d\tLoss: %.6f\n", e, L);
            }
        }

        // Predicciones finales
        double[][] yPred = model.forward(X);
        System.out.println("\nPredicciones XOR:");
        for (int i = 0; i < X.length; i++) {
            System.out.printf("%s XOR %s -> %.4f (clas=%d)\n",
                    (int)X[i][0], (int)X[i][1], yPred[i][0], (yPred[i][0] > 0.5 ? 1 : 0));
        }
    }
}
