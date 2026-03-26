package catboost.training.tree;

import catboost.training.QuantizedDataset;

public class HistogramBuilder {

    public Histogram build(QuantizedDataset dataset,
                           int featureIndex,
                           int leafCount,
                           int[] rowLeafIndexes,
                           double[] gradients,
                           double[] hessians) {
        int binCount = dataset.getBorderCount(featureIndex) + 1;
        Histogram histogram = new Histogram(leafCount, binCount);
        short[] bins = dataset.getBinsForFeature(featureIndex);
        for (int row = 0; row < dataset.getRowCount(); row++) {
            histogram.add(rowLeafIndexes[row], bins[row], gradients[row], hessians[row]);
        }
        return histogram;
    }

    public SplitCandidate findBestSplit(int featureIndex, Histogram histogram, double l2LeafReg) {
        double[][] gradientSums = histogram.getGradientSums();
        double[][] hessianSums = histogram.getHessianSums();
        int leafCount = gradientSums.length;
        int binCount = gradientSums[0].length;
        if (binCount <= 1) {
            return null;
        }

        double[] totalGradient = new double[leafCount];
        double[] totalHessian = new double[leafCount];
        for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
            for (int binIndex = 0; binIndex < binCount; binIndex++) {
                totalGradient[leafIndex] += gradientSums[leafIndex][binIndex];
                totalHessian[leafIndex] += hessianSums[leafIndex][binIndex];
            }
        }

        double[] leftGradient = new double[leafCount];
        double[] leftHessian = new double[leafCount];
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestBorderIndex = -1;

        for (int borderIndex = 0; borderIndex < binCount - 1; borderIndex++) {
            double score = 0.0;
            for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
                leftGradient[leafIndex] += gradientSums[leafIndex][borderIndex];
                leftHessian[leafIndex] += hessianSums[leafIndex][borderIndex];

                double rightGradient = totalGradient[leafIndex] - leftGradient[leafIndex];
                double rightHessian = totalHessian[leafIndex] - leftHessian[leafIndex];
                score += leafScore(leftGradient[leafIndex], leftHessian[leafIndex], l2LeafReg)
                        + leafScore(rightGradient, rightHessian, l2LeafReg);
            }

            if (score > bestScore) {
                bestScore = score;
                bestBorderIndex = borderIndex;
            }
        }

        if (bestBorderIndex < 0) {
            return null;
        }
        return new SplitCandidate(featureIndex, bestBorderIndex, bestScore);
    }

    private double leafScore(double gradient, double hessian, double l2LeafReg) {
        if (hessian <= 0.0) {
            return 0.0;
        }
        return (gradient * gradient) / (hessian + l2LeafReg);
    }
}
