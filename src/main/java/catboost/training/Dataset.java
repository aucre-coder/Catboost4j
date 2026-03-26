package catboost.training;

import java.util.List;

/**
 * Float-only training dataset for regression.
 */
public class Dataset {

    private final double[][] floatFeatures;
    private final double[] targets;
    private final double[] weights;
    private final FeatureSchema featureSchema;

    private Dataset(double[][] floatFeatures, double[] targets, double[] weights, FeatureSchema featureSchema) {
        validate(floatFeatures, targets, weights, featureSchema);
        this.floatFeatures = copy(floatFeatures);
        this.targets = copy(targets);
        this.weights = weights == null ? null : copy(weights);
        this.featureSchema = featureSchema;
    }

    public static Dataset of(double[][] floatFeatures, double[] targets, List<String> featureNames) {
        return new Dataset(floatFeatures, targets, null, new FeatureSchema(featureNames));
    }

    public static Dataset of(double[][] floatFeatures, double[] targets, double[] weights, List<String> featureNames) {
        return new Dataset(floatFeatures, targets, weights, new FeatureSchema(featureNames));
    }

    private static void validate(double[][] floatFeatures, double[] targets, double[] weights, FeatureSchema featureSchema) {
        if (floatFeatures == null || floatFeatures.length == 0) {
            throw new IllegalArgumentException("floatFeatures must not be empty");
        }
        if (targets == null || targets.length != floatFeatures.length) {
            throw new IllegalArgumentException("targets length must match row count");
        }
        if (weights != null && weights.length != floatFeatures.length) {
            throw new IllegalArgumentException("weights length must match row count");
        }
        if (weights != null) {
            for (int i = 0; i < weights.length; i++) {
                if (weights[i] <= 0.0) {
                    throw new IllegalArgumentException("weights must be positive");
                }
            }
        }
        if (featureSchema.size() != floatFeatures[0].length) {
            throw new IllegalArgumentException("feature count must match feature names");
        }
        for (int row = 0; row < floatFeatures.length; row++) {
            if (floatFeatures[row] == null || floatFeatures[row].length != featureSchema.size()) {
                throw new IllegalArgumentException("all rows must have the same width");
            }
        }
    }

    private static double[][] copy(double[][] values) {
        double[][] copy = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            copy[i] = copy(values[i]);
        }
        return copy;
    }

    private static double[] copy(double[] values) {
        double[] copy = new double[values.length];
        System.arraycopy(values, 0, copy, 0, values.length);
        return copy;
    }

    public int getRowCount() {
        return floatFeatures.length;
    }

    public int getFeatureCount() {
        return featureSchema.size();
    }

    public double getFeatureValue(int rowIndex, int featureIndex) {
        return floatFeatures[rowIndex][featureIndex];
    }

    public double[] getTargets() {
        return copy(targets);
    }

    public double[] getWeights() {
        return weights == null ? null : copy(weights);
    }

    public FeatureSchema getFeatureSchema() {
        return featureSchema;
    }

    public double[][] getFloatFeatures() {
        return copy(floatFeatures);
    }
}
