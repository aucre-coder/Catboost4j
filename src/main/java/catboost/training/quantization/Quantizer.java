package catboost.training.quantization;

import catboost.training.Dataset;
import catboost.training.QuantizedDataset;
import catboost.training.TrainerConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
        List<Double> borders = new ArrayList<Double>();
        for (int borderIndex = 1; borderIndex <= maxBorders; borderIndex++) {
            int splitPosition = (int) Math.floor((double) borderIndex * sorted.length / (maxBorders + 1));
            if (splitPosition <= 0 || splitPosition >= sorted.length) {
                continue;
            }
            double left = sorted[splitPosition - 1];
            double right = sorted[splitPosition];
            if (right <= left) {
                continue;
            }
            double border = left + ((right - left) / 2.0);
            if (borders.isEmpty() || border > borders.get(borders.size() - 1)) {
                borders.add(border);
            }
        }
        double[] result = new double[borders.size()];
        for (int i = 0; i < borders.size(); i++) {
            result[i] = borders.get(i);
        }
        return result;
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
}
