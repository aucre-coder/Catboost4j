package catboost.training.tree;

import catboost.training.ObliviousSplit;
import catboost.training.ObliviousTree;
import catboost.training.QuantizedDataset;
import catboost.training.TrainerConfig;

import java.util.ArrayList;
import java.util.List;

public class ObliviousTreeBuilder {

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
        for (int level = 0; level < config.getDepth(); level++) {
            int leafCount = 1 << level;
            SplitCandidate bestCandidate = null;
            for (int featureIndex = 0; featureIndex < dataset.getFeatureCount(); featureIndex++) {
                if (dataset.getBorderCount(featureIndex) == 0) {
                    continue;
                }
                Histogram histogram = histogramBuilder.build(dataset, featureIndex, leafCount, rowLeafIndexes, gradients, hessians);
                SplitCandidate candidate = histogramBuilder.findBestSplit(featureIndex, histogram, config.getL2LeafReg());
                if (candidate != null && (bestCandidate == null || candidate.getScore() > bestCandidate.getScore())) {
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate == null) {
                break;
            }

            short[] bins = dataset.getBinsForFeature(bestCandidate.getFeatureIndex());
            int splitMask = 1 << level;
            for (int row = 0; row < dataset.getRowCount(); row++) {
                if (bins[row] > bestCandidate.getBorderIndex()) {
                    rowLeafIndexes[row] |= splitMask;
                }
            }
            splits.add(new ObliviousSplit(bestCandidate.getFeatureIndex(), bestCandidate.getBorderIndex()));
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
        for (int leafIndex = 0; leafIndex < leafCount; leafIndex++) {
            leafValues[leafIndex] = (-config.getLearningRate() * leafGradientSums[leafIndex])
                    / (leafHessianSums[leafIndex] + config.getL2LeafReg());
        }

        return new ObliviousTree(splits, leafValues);
    }
}
