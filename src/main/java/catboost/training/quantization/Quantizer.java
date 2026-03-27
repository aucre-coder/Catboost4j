package catboost.training.quantization;

import catboost.training.Dataset;
import catboost.training.QuantizedDataset;
import catboost.training.TrainerConfig;

import java.util.Arrays;
import java.util.PriorityQueue;

public class Quantizer {

    public QuantizedDataset fit(Dataset dataset, TrainerConfig config) {
        int featureCount = dataset.getFeatureCount();
        int rowCount = dataset.getRowCount();
        short[][] bins = new short[featureCount][rowCount];
        double[][] borders = new double[featureCount][];

        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            double[] featureValues = new double[rowCount];
            for (int row = 0; row < rowCount; row++) {
                featureValues[row] = dataset.getFeatureValue(row, featureIndex);
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

    static double[] buildBorders(double[] values, int maxBins) {
        double[] sorted = copy(values);
        Arrays.sort(sorted);
        int maxBorders = Math.min(maxBins - 1, sorted.length - 1);
        if (maxBorders <= 0 || sorted[0] == sorted[sorted.length - 1]) {
            return new double[0];
        }

        PriorityQueue<GreedyBin> bins = new PriorityQueue<GreedyBin>();
        bins.add(new GreedyBin(sorted, 0, sorted.length));
        while (bins.size() <= maxBorders && bins.peek() != null && bins.peek().canSplit()) {
            GreedyBin bestBin = bins.poll();
            bins.add(bestBin.splitLeft());
            bins.add(bestBin);
        }

        double[] result = new double[bins.size() - 1];
        int index = 0;
        while (!bins.isEmpty()) {
            GreedyBin bin = bins.poll();
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

    private static double[] copy(double[] values) {
        double[] copy = new double[values.length];
        System.arraycopy(values, 0, copy, 0, values.length);
        return copy;
    }

    private static final class GreedyBin implements Comparable<GreedyBin> {
        private final double[] sortedValues;
        private int start;
        private final int end;
        private int bestSplit;
        private double bestScore;

        GreedyBin(double[] sortedValues, int start, int end) {
            this.sortedValues = sortedValues;
            this.start = start;
            this.end = end;
            updateBestSplit();
        }

        boolean canSplit() {
            return bestSplit > start && bestSplit < end;
        }

        boolean isFirst() {
            return start == 0;
        }

        GreedyBin splitLeft() {
            if (!canSplit()) {
                throw new IllegalStateException("bin cannot be split");
            }
            GreedyBin left = new GreedyBin(sortedValues, start, bestSplit);
            start = bestSplit;
            updateBestSplit();
            return left;
        }

        double leftBorder() {
            return (sortedValues[start - 1] + sortedValues[start]) * 0.5;
        }

        public int compareTo(GreedyBin other) {
            int scoreComparison = Double.compare(other.bestScore, bestScore);
            if (scoreComparison != 0) {
                return scoreComparison;
            }
            int sizeComparison = Integer.compare(other.end - other.start, end - start);
            if (sizeComparison != 0) {
                return sizeComparison;
            }
            return Integer.compare(start, other.start);
        }

        private void updateBestSplit() {
            if (end - start <= 1) {
                bestSplit = start;
                bestScore = Double.NEGATIVE_INFINITY;
                return;
            }

            int middle = start + ((end - start) / 2);
            double middleValue = sortedValues[middle];
            int lowerBound = lowerBound(sortedValues, start, middle, middleValue);
            int upperBound = upperBound(sortedValues, middle, end, middleValue);

            double leftScore = splitScore(lowerBound);
            double rightScore = splitScore(upperBound);
            bestSplit = leftScore >= rightScore ? lowerBound : upperBound;
            bestScore = leftScore >= rightScore ? leftScore : rightScore;
        }

        private double splitScore(int splitPosition) {
            if (splitPosition <= start || splitPosition >= end) {
                return Double.NEGATIVE_INFINITY;
            }
            return Math.log(splitPosition - start) + Math.log(end - splitPosition) - Math.log(end - start);
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

        private static int upperBound(double[] values, int from, int to, double target) {
            int left = from;
            int right = to;
            while (left < right) {
                int middle = left + ((right - left) / 2);
                if (values[middle] <= target) {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            return left;
        }
    }
}
