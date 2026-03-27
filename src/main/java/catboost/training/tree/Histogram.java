package catboost.training.tree;

public class Histogram {

    private final double[][] gradientSums;
    private final double[][] hessianSums;
    private final double[] splitScores;

    public Histogram(int leafCount, int binCount) {
        this.gradientSums = new double[leafCount][binCount];
        this.hessianSums = new double[leafCount][binCount];
        this.splitScores = binCount <= 1 ? new double[0] : new double[binCount - 1];
    }

    public void add(int leafIndex, int binIndex, double gradient, double hessian) {
        gradientSums[leafIndex][binIndex] += gradient;
        hessianSums[leafIndex][binIndex] += hessian;
    }

    public void addSplitScore(int borderIndex, double score) {
        splitScores[borderIndex] += score;
    }

    public double[][] getGradientSums() {
        return gradientSums;
    }

    public double[][] getHessianSums() {
        return hessianSums;
    }

    public double[] getSplitScores() {
        return splitScores;
    }
}
