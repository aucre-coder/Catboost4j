package catboost.training;

import catboost.model.Model;
import catboost.training.adapter.ModelAdapter;
import catboost.training.loss.LossFunction;
import catboost.training.loss.RMSELoss;
import catboost.training.quantization.Quantizer;
import catboost.training.tree.ObliviousTreeBuilder;

import java.util.ArrayList;
import java.util.List;

public class CatBoostTrainer {

    private final TrainerConfig config;
    private final Quantizer quantizer;
    private final LossFunction lossFunction;
    private final ObliviousTreeBuilder treeBuilder;
    private final ModelAdapter modelAdapter;

    public CatBoostTrainer() {
        this(new TrainerConfig());
    }

    public CatBoostTrainer(TrainerConfig config) {
        this(config, new Quantizer(), new RMSELoss(), new ObliviousTreeBuilder(), new ModelAdapter());
    }

    CatBoostTrainer(TrainerConfig config,
                    Quantizer quantizer,
                    LossFunction lossFunction,
                    ObliviousTreeBuilder treeBuilder,
                    ModelAdapter modelAdapter) {
        this.config = config;
        this.quantizer = quantizer;
        this.lossFunction = lossFunction;
        this.treeBuilder = treeBuilder;
        this.modelAdapter = modelAdapter;
    }

    public Model fit(Dataset dataset) {
        return modelAdapter.toModel(fitResult(dataset));
    }

    public TrainingResult fitResult(Dataset dataset) {
        QuantizedDataset quantizedDataset = quantizer.fit(dataset, config);
        double[] targets = dataset.getTargets();
        double[] weights = dataset.getWeights();

        double bias = computeWeightedMean(targets, weights);
        double[] predictions = new double[dataset.getRowCount()];
        for (int i = 0; i < predictions.length; i++) {
            predictions[i] = bias;
        }

        double[] gradients = new double[dataset.getRowCount()];
        double[] hessians = new double[dataset.getRowCount()];
        int[] rowLeafIndexes = new int[dataset.getRowCount()];
        List<ObliviousTree> trees = new ArrayList<ObliviousTree>();
        List<Double> losses = new ArrayList<Double>();
        losses.add(lossFunction.computeLoss(predictions, targets, weights));

        for (int iteration = 0; iteration < config.getIterations(); iteration++) {
            lossFunction.computeGradients(predictions, targets, weights, gradients, hessians);
            ObliviousTree tree = treeBuilder.build(quantizedDataset, gradients, hessians, config, rowLeafIndexes);
            trees.add(tree);
            for (int row = 0; row < predictions.length; row++) {
                predictions[row] += tree.getLeafValue(rowLeafIndexes[row]);
            }
            losses.add(lossFunction.computeLoss(predictions, targets, weights));
        }

        return new TrainingResult(trees, dataset.getFeatureSchema(), extractBorders(quantizedDataset), bias, losses);
    }

    private double computeWeightedMean(double[] values, double[] weights) {
        double weightedSum = 0.0;
        double totalWeight = 0.0;
        for (int i = 0; i < values.length; i++) {
            double weight = weights == null ? 1.0 : weights[i];
            weightedSum += values[i] * weight;
            totalWeight += weight;
        }
        return weightedSum / totalWeight;
    }

    private double[][] extractBorders(QuantizedDataset quantizedDataset) {
        double[][] borders = new double[quantizedDataset.getFeatureCount()][];
        for (int featureIndex = 0; featureIndex < quantizedDataset.getFeatureCount(); featureIndex++) {
            double[] source = quantizedDataset.getBorders(featureIndex);
            borders[featureIndex] = new double[source.length];
            System.arraycopy(source, 0, borders[featureIndex], 0, source.length);
        }
        return borders;
    }
}
