package catboost.training.tree;

import catboost.training.IterationContext;
import catboost.training.ObliviousSplit;
import catboost.training.ObliviousTree;
import catboost.training.OrderedTrainingState;
import catboost.training.QuantizedDataset;
import catboost.training.TrainingDebugLog;
import catboost.training.TrainerConfig;

import java.util.ArrayList;
import java.util.List;

public class ObliviousTreeBuilder {

    public static final class TreeBuildResult {
        private final ObliviousTree tree;
        private final double[] trainingRowUpdates;
        private final double[][][] learnFoldBodyTailLeafValues;

        TreeBuildResult(ObliviousTree tree, double[] trainingRowUpdates, double[][][] learnFoldBodyTailLeafValues) {
            this.tree = tree;
            this.trainingRowUpdates = copy(trainingRowUpdates);
            this.learnFoldBodyTailLeafValues = copy(learnFoldBodyTailLeafValues);
        }

        public ObliviousTree getTree() {
            return tree;
        }

        public double[] getTrainingRowUpdates() {
            return copy(trainingRowUpdates);
        }

        public double[] getLearnFoldBodyTailLeafValues(int foldIndex, int bodyTailIndex) {
            return copy(learnFoldBodyTailLeafValues[foldIndex][bodyTailIndex]);
        }

        private static double[] copy(double[] values) {
            double[] copy = new double[values.length];
            System.arraycopy(values, 0, copy, 0, values.length);
            return copy;
        }

        private static double[][][] copy(double[][][] values) {
            double[][][] copy = new double[values.length][][];
            for (int foldIndex = 0; foldIndex < values.length; foldIndex++) {
                copy[foldIndex] = new double[values[foldIndex].length][];
                for (int bodyTailIndex = 0; bodyTailIndex < values[foldIndex].length; bodyTailIndex++) {
                    copy[foldIndex][bodyTailIndex] = copy(values[foldIndex][bodyTailIndex]);
                }
            }
            return copy;
        }
    }

    private final HistogramBuilder histogramBuilder;

    public ObliviousTreeBuilder() {
        this(new HistogramBuilder());
    }

    public ObliviousTreeBuilder(HistogramBuilder histogramBuilder) {
        this.histogramBuilder = histogramBuilder;
    }

    public ObliviousTree build(QuantizedDataset dataset,
                               double[] gradients,
                               double[] hessians,
                               TrainerConfig config,
                               int[] rowLeafIndexes) {
        for (int row = 0; row < rowLeafIndexes.length; row++) {
            rowLeafIndexes[row] = 0;
        }

        List<ObliviousSplit> splits = new ArrayList<ObliviousSplit>();
        boolean[][] usedSplits = allocateUsedSplits(dataset);
        for (int level = 0; level < config.getDepth(); level++) {
            int leafCount = 1 << level;
            SplitCandidate bestCandidate = null;
            for (int featureIndex = 0; featureIndex < dataset.getFeatureCount(); featureIndex++) {
                if (dataset.getBorderCount(featureIndex) == 0) {
                    continue;
                }
                Histogram histogram = histogramBuilder.build(dataset, featureIndex, leafCount, rowLeafIndexes, gradients, hessians);
                SplitCandidate candidate = histogramBuilder.findBestSplit(featureIndex, histogram, usedSplits[featureIndex]);
                if (candidate != null && (bestCandidate == null || candidate.getScore() > bestCandidate.getScore())) {
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate == null) {
                TrainingDebugLog.log("iterationTree level=%d no candidate", level);
                break;
            }

            applySplit(dataset, bestCandidate, rowLeafIndexes, level);
            splits.add(new ObliviousSplit(bestCandidate.getFeatureIndex(), bestCandidate.getBorderIndex()));
            markUsed(bestCandidate, usedSplits);
            TrainingDebugLog.log(
                    "iterationTree level=%d choose feature=%d border=%d score=%.12f rowLeafIndexes=%s",
                    level,
                    bestCandidate.getFeatureIndex(),
                    bestCandidate.getBorderIndex(),
                    bestCandidate.getScore(),
                    TrainingDebugLog.formatIntArray(rowLeafIndexes)
            );
        }

        int leafCount = 1 << splits.size();
        double[] leafGradientSums = new double[leafCount];
        double[] leafHessianSums = new double[leafCount];
        for (int row = 0; row < dataset.getRowCount(); row++) {
            int leafIndex = rowLeafIndexes[row];
            leafGradientSums[leafIndex] += gradients[row];
            leafHessianSums[leafIndex] += hessians[row];
        }

        double[] leafValues = new double[leafCount];
        double scaledL2Regularizer = scaledL2Regularizer(config.getL2LeafReg(), sum(hessians), hessians.length);
        for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
            leafValues[leafIndex] = (config.getLearningRate() * leafGradientSums[leafIndex])
                    / (leafHessianSums[leafIndex] + scaledL2Regularizer);
        }
        return new ObliviousTree(splits, leafValues);
    }

    public TreeBuildResult buildOrdered(QuantizedDataset dataset,
                                        OrderedTrainingState trainingState,
                                        IterationContext iterationContext,
                                        TrainerConfig config,
                                        int[] rowLeafIndexes) {
        OrderedTrainingState.LearnFold selectedLearnFold = trainingState.getLearnFold(iterationContext.getSelectedLearnFoldIndex());
        double[] bootstrapWeights = iterationContext.getBootstrapWeights();
        for (int row = 0; row < rowLeafIndexes.length; row++) {
            rowLeafIndexes[row] = 0;
        }

        List<ObliviousSplit> splits = new ArrayList<ObliviousSplit>();
        boolean[][] usedSplits = allocateUsedSplits(dataset);
        for (int level = 0; level < config.getDepth(); level++) {
            int leafCount = 1 << level;
            SplitCandidate bestCandidate = null;
            for (int featureIndex = 0; featureIndex < dataset.getFeatureCount(); featureIndex++) {
                if (dataset.getBorderCount(featureIndex) == 0) {
                    continue;
                }
                SplitCandidate candidate = histogramBuilder.findBestOrderedSplit(
                        dataset,
                        featureIndex,
                        leafCount,
                        rowLeafIndexes,
                        selectedLearnFold,
                        bootstrapWeights,
                        trainingState.getRandom(),
                        iterationContext.getScoreStdDev(),
                        config,
                        usedSplits[featureIndex]
                );
                if (candidate != null && (bestCandidate == null || candidate.getScore() > bestCandidate.getScore())) {
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate == null) {
                break;
            }

            applySplit(dataset, bestCandidate, rowLeafIndexes, level);
            splits.add(new ObliviousSplit(bestCandidate.getFeatureIndex(), bestCandidate.getBorderIndex()));
            markUsed(bestCandidate, usedSplits);
        }

        int leafCount = 1 << splits.size();
        double[] averagingLeafValues = buildAveragingLeafValues(trainingState.getAveragingFold(), rowLeafIndexes, leafCount, bootstrapWeights, config);
        double[] trainingRowUpdates = buildRowUpdates(rowLeafIndexes, averagingLeafValues);
        TrainingDebugLog.log("iterationTree finalRowLeafIndexes=%s", TrainingDebugLog.formatIntArray(rowLeafIndexes));
        TrainingDebugLog.log("iterationTree averagingLeafValues=%s", TrainingDebugLog.formatDoubleArray(averagingLeafValues));

        double[][][] learnFoldBodyTailLeafValues = new double[trainingState.getLearnFoldCount()][][];
        for (int foldIndex = 0; foldIndex < trainingState.getLearnFoldCount(); foldIndex++) {
            OrderedTrainingState.LearnFold learnFold = trainingState.getLearnFold(foldIndex);
            learnFoldBodyTailLeafValues[foldIndex] = new double[learnFold.getBodyTailCount()][];
            for (int bodyTailIndex = 0; bodyTailIndex < learnFold.getBodyTailCount(); bodyTailIndex++) {
                learnFoldBodyTailLeafValues[foldIndex][bodyTailIndex] = buildLearnBodyTailLeafValues(
                        learnFold,
                        learnFold.getBodyTail(bodyTailIndex),
                        rowLeafIndexes,
                        leafCount,
                        bootstrapWeights,
                        config
                );
                TrainingDebugLog.log(
                        "iterationTree learnFold=%d bodyTail=%d leafValues=%s",
                        foldIndex,
                        bodyTailIndex,
                        TrainingDebugLog.formatDoubleArray(learnFoldBodyTailLeafValues[foldIndex][bodyTailIndex])
                );
            }
        }

        return new TreeBuildResult(new ObliviousTree(splits, averagingLeafValues), trainingRowUpdates, learnFoldBodyTailLeafValues);
    }

    private double[] buildAveragingLeafValues(OrderedTrainingState.AveragingFold averagingFold,
                                              int[] rowLeafIndexes,
                                              int leafCount,
                                              double[] bootstrapWeights,
                                              TrainerConfig config) {
        double[] leafGradientSums = new double[leafCount];
        double[] leafWeightSums = new double[leafCount];
        double[] predictions = averagingFold.getPredictions();
        double[] targets = averagingFold.getTargets();
        double[] weights = averagingFold.getWeights();
        for (int row = 0; row < rowLeafIndexes.length; row++) {
            double weight = weights == null ? 1.0 : weights[row];
            weight *= bootstrapWeights[row];
            int leafIndex = rowLeafIndexes[row];
            leafGradientSums[leafIndex] += (targets[row] - predictions[row]) * weight;
            leafWeightSums[leafIndex] += weight;
        }

        double totalWeight = sumWeights(weights, bootstrapWeights, rowLeafIndexes.length);
        double scaledL2 = scaledL2Regularizer(config.getL2LeafReg(), totalWeight, rowLeafIndexes.length);
        double[] leafValues = new double[leafCount];
        for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
            leafValues[leafIndex] = (config.getLearningRate() * leafGradientSums[leafIndex])
                    / (leafWeightSums[leafIndex] + scaledL2);
        }
        return leafValues;
    }

    private double[] buildLearnBodyTailLeafValues(OrderedTrainingState.LearnFold learnFold,
                                                  OrderedTrainingState.BodyTail bodyTail,
                                                  int[] rowLeafIndexes,
                                                  int leafCount,
                                                  double[] bootstrapWeights,
                                                  TrainerConfig config) {
        double[] leafGradientSums = new double[leafCount];
        double[] leafWeightSums = new double[leafCount];
        int[] orderedToOriginal = learnFold.getOrderedToOriginal();
        double[] orderedTargets = learnFold.getOrderedTargets();
        double[] orderedWeights = learnFold.getOrderedWeights();
        double[] approximations = bodyTail.getApproximations();
        for (int orderedIndex = 0; orderedIndex < bodyTail.getBodyFinish(); orderedIndex++) {
            int row = orderedToOriginal[orderedIndex];
            double weight = orderedWeights == null ? 1.0 : orderedWeights[orderedIndex];
            weight *= bootstrapWeights[row];
            int leafIndex = rowLeafIndexes[row];
            leafGradientSums[leafIndex] += (orderedTargets[orderedIndex] - approximations[orderedIndex]) * weight;
            leafWeightSums[leafIndex] += weight;
        }

        double scaledL2 = scaledL2Regularizer(
                config.getL2LeafReg(),
                bodyTail.getBodySumWeight(),
                bodyTail.getBodyFinish()
        );
        double[] leafValues = new double[leafCount];
        for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
            leafValues[leafIndex] = (config.getLearningRate() * leafGradientSums[leafIndex])
                    / (leafWeightSums[leafIndex] + scaledL2);
        }
        return leafValues;
    }

    private double[] buildRowUpdates(int[] rowLeafIndexes, double[] leafValues) {
        double[] updates = new double[rowLeafIndexes.length];
        for (int row = 0; row < rowLeafIndexes.length; row++) {
            updates[row] = leafValues[rowLeafIndexes[row]];
        }
        return updates;
    }

    private void applySplit(QuantizedDataset dataset, SplitCandidate splitCandidate, int[] rowLeafIndexes, int level) {
        short[] bins = dataset.getBinsForFeature(splitCandidate.getFeatureIndex());
        int splitMask = 1 << level;
        for (int row = 0; row < dataset.getRowCount(); row++) {
            if (bins[row] > splitCandidate.getBorderIndex()) {
                rowLeafIndexes[row] |= splitMask;
            }
        }
    }

    private double scaledL2Regularizer(double l2LeafReg, double sumWeight, int docCount) {
        if (docCount <= 0) {
            return 0.0;
        }
        return l2LeafReg * (sumWeight / docCount);
    }

    private double sum(double[] values) {
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i];
        }
        return sum;
    }

    private double sumWeights(double[] weights, double[] bootstrapWeights, int rowCount) {
        if (weights == null) {
            double sum = 0.0;
            for (int row = 0; row < rowCount; row++) {
                sum += bootstrapWeights[row];
            }
            return sum;
        }
        double sum = 0.0;
        for (int row = 0; row < weights.length; row++) {
            sum += weights[row] * bootstrapWeights[row];
        }
        return sum;
    }

    private boolean[][] allocateUsedSplits(QuantizedDataset dataset) {
        boolean[][] usedSplits = new boolean[dataset.getFeatureCount()][];
        for (int featureIndex = 0; featureIndex < dataset.getFeatureCount(); featureIndex++) {
            usedSplits[featureIndex] = new boolean[dataset.getBorderCount(featureIndex)];
        }
        return usedSplits;
    }

    private void markUsed(SplitCandidate candidate, boolean[][] usedSplits) {
        usedSplits[candidate.getFeatureIndex()][candidate.getBorderIndex()] = true;
    }
}
