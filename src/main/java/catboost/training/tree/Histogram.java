package catboost.training.tree;

public class Histogram {

    private final double[][] gradientSums;
    private final double[][] hessianSums;

    public Histogram(int leafCount, int binCount) {
        this.gradientSums = new double[leafCount][binCount];
        this.hessianSums = new double[leafCount][binCount];
    }

    public void add(int leafIndex, int binIndex, double gradient, double hessian) {
        gradientSums[leafIndex][binIndex] += gradient;
        hessianSums[leafIndex][binIndex] += hessian;
    }

    public double[][] getGradientSums() {
        return gradientSums;
    }

    public double[][] getHessianSums() {
        return hessianSums;
    }
}
