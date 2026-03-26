package catboost.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TrainingResult {

    private final List<ObliviousTree> trees;
    private final FeatureSchema featureSchema;
    private final double[][] borders;
    private final double bias;
    private final List<Double> trainingLosses;

    public TrainingResult(List<ObliviousTree> trees,
                          FeatureSchema featureSchema,
                          double[][] borders,
                          double bias,
                          List<Double> trainingLosses) {
        this.trees = Collections.unmodifiableList(new ArrayList<ObliviousTree>(trees));
        this.featureSchema = featureSchema;
        this.borders = copy(borders);
        this.bias = bias;
        this.trainingLosses = Collections.unmodifiableList(new ArrayList<Double>(trainingLosses));
    }

    public List<ObliviousTree> getTrees() {
        return trees;
    }

    public FeatureSchema getFeatureSchema() {
        return featureSchema;
    }

    public double[][] getBorders() {
        return copy(borders);
    }

    public double getBias() {
        return bias;
    }

    public List<Double> getTrainingLosses() {
        return trainingLosses;
    }

    private static double[][] copy(double[][] values) {
        double[][] copy = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            copy[i] = new double[values[i].length];
            System.arraycopy(values[i], 0, copy[i], 0, values[i].length);
        }
        return copy;
    }
}
