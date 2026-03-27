package catboost.training.tree;

import catboost.training.OrderedTrainingState;
import catboost.training.QuantizedDataset;
import catboost.training.TrainerConfig;

public class HistogramBuilder {

    public Histogram build(QuantizedDataset dataset,
                           int featureIndex,
                           int leafCount,
                           int[] rowLeafIndexes,
                           double[] gradients,
                           double[] hessians) {
        return build(dataset,
                featureIndex,
                leafCount,
                rowLeafIndexes,
                gradients,
                hessians,
                identityOrder(dataset.getRowCount()),
                0.0);
    }

    public Histogram build(QuantizedDataset dataset,
                           int featureIndex,
                           int leafCount,
                           int[] rowLeafIndexes,
                           double[] gradients,
                           double[] hessians,
                           int[] orderedToOriginal,
                           double scaledL2Regularizer) {
        int binCount = dataset.getBorderCount(featureIndex) + 1;
        Histogram histogram = new Histogram(leafCount, binCount);
        short[] bins = dataset.getBinsForFeature(featureIndex);
        for (int row = 0; row < dataset.getRowCount(); row++) {
            histogram.add(rowLeafIndexes[row], bins[row], gradients[row], hessians[row]);
        }

        if (binCount <= 1) {
            return histogram;
        }

        double[][] leftGradientSums = new double[leafCount][binCount - 1];
        double[][] leftWeightSums = new double[leafCount][binCount - 1];
        double[] totalGradientSums = new double[leafCount];
        double[] totalWeightSums = new double[leafCount];

        for (int orderedIndex = 0; orderedIndex < orderedToOriginal.length; orderedIndex++) {
            int row = orderedToOriginal[orderedIndex];
            int leafIndex = rowLeafIndexes[row];
            int binIndex = bins[row];
            double gradient = gradients[row];
            double hessian = hessians[row];

            for (int borderIndex = 0; borderIndex < binCount - 1; borderIndex++) {
                double sideGradient;
                double sideWeight;
                if (binIndex <= borderIndex) {
                    sideGradient = leftGradientSums[leafIndex][borderIndex];
                    sideWeight = leftWeightSums[leafIndex][borderIndex];
                } else {
                    sideGradient = totalGradientSums[leafIndex] - leftGradientSums[leafIndex][borderIndex];
                    sideWeight = totalWeightSums[leafIndex] - leftWeightSums[leafIndex][borderIndex];
                }

                if (sideWeight > 0.0) {
                    histogram.addSplitScore(borderIndex, (gradient * sideGradient) / (sideWeight + scaledL2Regularizer));
                }
            }

            totalGradientSums[leafIndex] += gradient;
            totalWeightSums[leafIndex] += hessian;
            for (int borderIndex = binIndex; borderIndex < binCount - 1; borderIndex++) {
                leftGradientSums[leafIndex][borderIndex] += gradient;
                leftWeightSums[leafIndex][borderIndex] += hessian;
            }
        }
        return histogram;
    }

    public SplitCandidate findBestSplit(int featureIndex, Histogram histogram, double l2LeafReg) {
        double[] splitScores = histogram.getSplitScores();
        if (splitScores.length == 0) {
            return null;
        }

        double bestScore = Double.NEGATIVE_INFINITY;
        int bestBorderIndex = -1;
        for (int borderIndex = 0; borderIndex < splitScores.length; borderIndex++) {
            double score = splitScores[borderIndex];
            if (score > bestScore) {
                bestScore = score;
                bestBorderIndex = borderIndex;
            }
        }

        if (bestBorderIndex < 0 || bestScore <= 0.0) {
            return null;
        }
        return new SplitCandidate(featureIndex, bestBorderIndex, bestScore);
    }

    public SplitCandidate findBestOrderedSplit(QuantizedDataset dataset,
                                               int featureIndex,
                                               int leafCount,
                                               int[] rowLeafIndexes,
                                               OrderedTrainingState.LearnFold learnFold,
                                               TrainerConfig config) {
        int borderCount = dataset.getBorderCount(featureIndex);
        if (borderCount == 0) {
            return null;
        }

        short[] bins = dataset.getBinsForFeature(featureIndex);
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestBorderIndex = -1;
        for (int borderIndex = 0; borderIndex < borderCount; borderIndex++) {
            double score = scoreOrderedSplit(bins, borderIndex, leafCount, rowLeafIndexes, learnFold, config);
            if (score > bestScore) {
                bestScore = score;
                bestBorderIndex = borderIndex;
            }
        }

        if (bestBorderIndex < 0 || !(bestScore > 0.0)) {
            return null;
        }
        return new SplitCandidate(featureIndex, bestBorderIndex, bestScore);
    }

    private double scoreOrderedSplit(short[] bins,
                                     int borderIndex,
                                     int leafCount,
                                     int[] rowLeafIndexes,
                                     OrderedTrainingState.LearnFold learnFold,
                                     TrainerConfig config) {
        double numerator = 0.0;
        double denominator = 1e-100;
        int[] orderedToOriginal = learnFold.getOrderedToOriginal();
        double[] orderedTargets = learnFold.getOrderedTargets();
        double[] orderedWeights = learnFold.getOrderedWeights();

        for (int bodyTailIndex = 0; bodyTailIndex < learnFold.getBodyTailCount(); bodyTailIndex++) {
            OrderedTrainingState.BodyTail bodyTail = learnFold.getBodyTail(bodyTailIndex);
            if (bodyTail.getTailFinish() <= bodyTail.getBodyFinish()) {
                continue;
            }

            double[] leftBodyGradients = new double[leafCount];
            double[] rightBodyGradients = new double[leafCount];
            double[] leftBodyWeights = new double[leafCount];
            double[] rightBodyWeights = new double[leafCount];
            double[] leftTailGradients = new double[leafCount];
            double[] rightTailGradients = new double[leafCount];
            double[] leftTailWeights = new double[leafCount];
            double[] rightTailWeights = new double[leafCount];

            double[] approximations = bodyTail.getApproximations();
            for (int orderedIndex = 0; orderedIndex < bodyTail.getBodyFinish(); orderedIndex++) {
                accumulateOrderedDoc(
                        bins,
                        borderIndex,
                        rowLeafIndexes,
                        orderedToOriginal,
                        orderedTargets,
                        orderedWeights,
                        approximations,
                        orderedIndex,
                        leftBodyGradients,
                        rightBodyGradients,
                        leftBodyWeights,
                        rightBodyWeights
                );
            }
            for (int orderedIndex = bodyTail.getBodyFinish(); orderedIndex < bodyTail.getTailFinish(); orderedIndex++) {
                accumulateOrderedDoc(
                        bins,
                        borderIndex,
                        rowLeafIndexes,
                        orderedToOriginal,
                        orderedTargets,
                        orderedWeights,
                        approximations,
                        orderedIndex,
                        leftTailGradients,
                        rightTailGradients,
                        leftTailWeights,
                        rightTailWeights
                );
            }

            double scaledL2 = scaledL2Regularizer(config.getL2LeafReg(), bodyTail.getBodySumWeight(), bodyTail.getBodyFinish());
            for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
                double leftValue = average(leftBodyGradients[leafIndex], leftBodyWeights[leafIndex], scaledL2);
                double rightValue = average(rightBodyGradients[leafIndex], rightBodyWeights[leafIndex], scaledL2);
                numerator += leftValue * leftTailGradients[leafIndex];
                numerator += rightValue * rightTailGradients[leafIndex];
                denominator += leftValue * leftValue * leftTailWeights[leafIndex];
                denominator += rightValue * rightValue * rightTailWeights[leafIndex];
            }
        }
        return numerator / Math.sqrt(denominator);
    }

    private void accumulateOrderedDoc(short[] bins,
                                      int borderIndex,
                                      int[] rowLeafIndexes,
                                      int[] orderedToOriginal,
                                      double[] orderedTargets,
                                      double[] orderedWeights,
                                      double[] approximations,
                                      int orderedIndex,
                                      double[] leftGradients,
                                      double[] rightGradients,
                                      double[] leftWeights,
                                      double[] rightWeights) {
        int row = orderedToOriginal[orderedIndex];
        int leafIndex = rowLeafIndexes[row];
        double weight = orderedWeights == null ? 1.0 : orderedWeights[orderedIndex];
        double gradient = (orderedTargets[orderedIndex] - approximations[orderedIndex]) * weight;
        if (bins[row] <= borderIndex) {
            leftGradients[leafIndex] += gradient;
            leftWeights[leafIndex] += weight;
        } else {
            rightGradients[leafIndex] += gradient;
            rightWeights[leafIndex] += weight;
        }
    }

    private double average(double sumGradient, double sumWeight, double scaledL2) {
        if (sumWeight <= 0.0) {
            return 0.0;
        }
        return sumGradient / (sumWeight + scaledL2);
    }

    private double scaledL2Regularizer(double l2LeafReg, double sumWeight, int docCount) {
        if (docCount <= 0) {
            return 0.0;
        }
        return l2LeafReg * (sumWeight / docCount);
    }

    private int[] identityOrder(int rowCount) {
        int[] orderedToOriginal = new int[rowCount];
        for (int row = 0; row < rowCount; row++) {
            orderedToOriginal[row] = row;
        }
        return orderedToOriginal;
    }
}
