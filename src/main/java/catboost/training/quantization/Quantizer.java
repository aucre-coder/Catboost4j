package catboost.training.quantization;

import catboost.training.Dataset;
import catboost.training.QuantizedDataset;
import catboost.training.TrainerConfig;

import java.util.Arrays;
import java.util.PriorityQueue;

public class Quantizer {

    private static final double LOG_EPS = 1e-8;

    public QuantizedDataset fit(Dataset dataset, TrainerConfig config) {
        int featureCount = dataset.getFeatureCount();
        int rowCount = dataset.getRowCount();
        short[][] bins = new short[featureCount][rowCount];
        double[][] borders = new double[featureCount][];

        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            double[] featureValues = new double[rowCount];
            for (int row = 0; row < rowCount; row++) {
                featureValues[row] = asCatBoostFloat(dataset.getFeatureValue(row, featureIndex));
                if (!Double.isFinite(featureValues[row])) {
                    throw new IllegalArgumentException("all feature values must be finite");
                }
            }

            double[] featureBorders = buildBorders(featureValues, config.getMaxBins());
            borders[featureIndex] = featureBorders;
            for (int row = 0; row < rowCount; row++) {
                bins[featureIndex][row] = (short) locateBin(featureValues[row], featureBorders);
            }
        }

        return new QuantizedDataset(bins, borders, dataset.getFeatureSchema(), rowCount);
    }

    public QuantizedDataset quantizeWithBorders(Dataset dataset, double[][] borders) {
        int featureCount = dataset.getFeatureCount();
        if (borders.length != featureCount) {
            throw new IllegalArgumentException("base borders feature count must match dataset feature count");
        }

        int rowCount = dataset.getRowCount();
        short[][] bins = new short[featureCount][rowCount];
        double[][] copiedBorders = new double[featureCount][];
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            double[] featureBorders = copy(borders[featureIndex]);
            copiedBorders[featureIndex] = featureBorders;
            for (int row = 0; row < rowCount; row++) {
                double value = asCatBoostFloat(dataset.getFeatureValue(row, featureIndex));
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("all feature values must be finite");
                }
                bins[featureIndex][row] = (short) locateBin(value, featureBorders);
            }
        }
        return new QuantizedDataset(bins, copiedBorders, dataset.getFeatureSchema(), rowCount);
    }

    static double[] buildBorders(double[] values, int maxBins) {
        double[] sorted = copy(values);
        Arrays.sort(sorted);
        int maxBorders = Math.min(maxBins - 1, sorted.length - 1);
        if (maxBorders <= 0 || sorted[0] == sorted[sorted.length - 1]) {
            return new double[0];
        }

        GroupedValues grouped = groupSortedValues(sorted);
        if (grouped.uniqueValues.length <= 1) {
            return new double[0];
        }

        maxBorders = Math.min(maxBorders, grouped.uniqueValues.length - 1);
        PriorityQueue<WeightedBin> bins = new PriorityQueue<WeightedBin>();
        bins.add(new WeightedBin(grouped.uniqueValues, grouped.cumulativeWeights, 0, grouped.uniqueValues.length));
        while (bins.size() <= maxBorders && bins.peek() != null && bins.peek().canSplit()) {
            WeightedBin bestBin = bins.poll();
            bins.add(bestBin.splitLeft());
            bins.add(bestBin);
        }

        double[] result = new double[bins.size() - 1];
        int index = 0;
        while (!bins.isEmpty()) {
            WeightedBin bin = bins.poll();
            if (!bin.isFirst()) {
                result[index++] = bin.leftBorder();
            }
        }
        Arrays.sort(result);
        if (index == result.length) {
            return result;
        }
        return Arrays.copyOf(result, index);
    }

    static int locateBin(double value, double[] borders) {
        int pos = Arrays.binarySearch(borders, value);
        if (pos >= 0) {
            return pos;
        }
        return -(pos + 1);
    }

    private static GroupedValues groupSortedValues(double[] sortedValues) {
        double[] uniqueValues = new double[sortedValues.length];
        double[] weights = new double[sortedValues.length];
        int uniqueCount = 0;

        for (int i = 0; i < sortedValues.length; i++) {
            double value = sortedValues[i];
            if (uniqueCount == 0 || Double.compare(uniqueValues[uniqueCount - 1], value) != 0) {
                uniqueValues[uniqueCount] = value;
                weights[uniqueCount] = 1.0;
                uniqueCount++;
            } else {
                weights[uniqueCount - 1] += 1.0;
            }
        }

        if (uniqueCount == 0) {
            return new GroupedValues(new double[0], new double[0]);
        }

        double[] compactValues = Arrays.copyOf(uniqueValues, uniqueCount);
        double[] cumulativeWeights = Arrays.copyOf(weights, uniqueCount);
        double totalWeight = cumulativeWeights[0];
        for (int i = 1; i < cumulativeWeights.length; i++) {
            totalWeight += cumulativeWeights[i];
            cumulativeWeights[i] += cumulativeWeights[i - 1];
        }

        double normalization = sortedValues.length / totalWeight;
        for (int i = 0; i < cumulativeWeights.length; i++) {
            cumulativeWeights[i] *= normalization;
        }
        return new GroupedValues(compactValues, cumulativeWeights);
    }

    private static double asCatBoostFloat(double value) {
        return (double) ((float) value);
    }

    private static double[] copy(double[] values) {
        double[] copy = new double[values.length];
        System.arraycopy(values, 0, copy, 0, values.length);
        return copy;
    }

    private static final class GroupedValues {
        private final double[] uniqueValues;
        private final double[] cumulativeWeights;

        private GroupedValues(double[] uniqueValues, double[] cumulativeWeights) {
            this.uniqueValues = uniqueValues;
            this.cumulativeWeights = cumulativeWeights;
        }
    }

    private static final class WeightedBin implements Comparable<WeightedBin> {
        private final double[] uniqueValues;
        private final double[] cumulativeWeights;
        private int start;
        private final int end;
        private int bestSplit;
        private double bestScore;

        private WeightedBin(double[] uniqueValues, double[] cumulativeWeights, int start, int end) {
            this.uniqueValues = uniqueValues;
            this.cumulativeWeights = cumulativeWeights;
            this.start = start;
            this.end = end;
            updateBestSplit();
        }

        private boolean canSplit() {
            return bestSplit > start && bestSplit < end;
        }

        private boolean isFirst() {
            return start == 0;
        }

        private WeightedBin splitLeft() {
            if (!canSplit()) {
                throw new IllegalStateException("bin cannot be split");
            }
            WeightedBin left = new WeightedBin(uniqueValues, cumulativeWeights, start, bestSplit);
            start = bestSplit;
            updateBestSplit();
            return left;
        }

        private double leftBorder() {
            return (uniqueValues[start - 1] * 0.5) + (uniqueValues[start] * 0.5);
        }

        public int compareTo(WeightedBin other) {
            return Double.compare(other.bestScore, bestScore);
        }

        private void updateBestSplit() {
            if (end - start <= 1) {
                bestSplit = start;
                bestScore = Double.NEGATIVE_INFINITY;
                return;
            }

            double leftBinsWeight = start == 0 ? 0.0 : cumulativeWeights[start - 1];
            double midCumulativeWeight = 0.5 * (leftBinsWeight + cumulativeWeights[end - 1]);
            int lowerBound = lowerBound(cumulativeWeights, start, end, midCumulativeWeight);
            if (lowerBound >= end) {
                lowerBound = end - 1;
            }
            int upperBound = lowerBound + 1;

            double leftScore = splitScore(lowerBound);
            double rightScore = splitScore(upperBound);
            bestSplit = leftScore >= rightScore ? lowerBound : upperBound;
            bestScore = leftScore >= rightScore ? leftScore : rightScore;
        }

        private double splitScore(int splitPosition) {
            if (splitPosition <= start || splitPosition >= end) {
                return Double.NEGATIVE_INFINITY;
            }
            double leftBinsWeight = start == 0 ? 0.0 : cumulativeWeights[start - 1];
            double leftPartWeight = cumulativeWeights[splitPosition - 1] - leftBinsWeight;
            double rightPartWeight = cumulativeWeights[end - 1] - cumulativeWeights[splitPosition - 1];
            return Math.log(leftPartWeight + LOG_EPS)
                    + Math.log(rightPartWeight + LOG_EPS)
                    - Math.log(leftPartWeight + rightPartWeight + LOG_EPS);
        }

        private static int lowerBound(double[] values, int from, int to, double target) {
            int left = from;
            int right = to;
            while (left < right) {
                int middle = left + ((right - left) / 2);
                if (values[middle] < target) {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            return left;
        }
    }
}
