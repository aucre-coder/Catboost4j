package catboost.training;

import catboost.model.Model;
import catboost.training.adapter.ModelAdapter;
import catboost.training.loss.LossFunction;
import catboost.training.loss.RMSELoss;
import catboost.training.quantization.Quantizer;
import catboost.training.tree.ObliviousTreeBuilder;

import java.util.ArrayList;
import java.util.Arrays;
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

    public Model fit(Dataset dataset, TrainingResult baseResult) {
        if (baseResult == null) {
            return fit(dataset);
        }
        return modelAdapter.toModel(fitResult(dataset, baseResult));
    }

    public TrainingResult fitResult(Dataset dataset) {
        return fitResult(dataset, null);
    }

    public TrainingResult fitResult(Dataset dataset, TrainingResult baseResult) {
        config.validateForParityScope();
        if (baseResult == null) {
            return fitCold(dataset);
        }
        return fitWarm(dataset, baseResult);
    }

    private TrainingResult fitCold(Dataset dataset) {
        QuantizedDataset quantizedDataset = quantizer.fit(dataset, config);
        double[] targets = dataset.getTargets();
        double[] weights = dataset.getWeights();
        double bias = computeWeightedMean(targets, weights);
        OrderedTrainingState trainingState = new OrderedTrainingState(dataset, bias, config.getRandomSeed());
        TrainingRunResult trainingRun = train(dataset, quantizedDataset, trainingState, bias);
        CompactedModel compactedModel = compactModel(trainingRun.trees, quantizedDataset);
        return new TrainingResult(compactedModel.trees, dataset.getFeatureSchema(), compactedModel.borders, bias, trainingRun.losses);
    }

    private TrainingResult fitWarm(Dataset dataset, TrainingResult baseResult) {
        validateWarmStartCompatibility(dataset, baseResult);
        double[][] baseBorders = baseResult.getBorders();
        QuantizedDataset quantizedDataset = quantizer.quantizeWithBorders(dataset, baseBorders);
        double[] initialPredictions = TrainingResultScorer.score(baseResult, dataset);
        OrderedTrainingState trainingState = new OrderedTrainingState(
                dataset,
                baseResult.getBias(),
                config.getRandomSeed(),
                initialPredictions
        );
        TrainingRunResult trainingRun = train(dataset, quantizedDataset, trainingState, baseResult.getBias());
        List<ObliviousTree> mergedTrees = new ArrayList<ObliviousTree>(baseResult.getTrees().size() + trainingRun.trees.size());
        mergedTrees.addAll(baseResult.getTrees());
        mergedTrees.addAll(trainingRun.trees);
        return new TrainingResult(
                mergedTrees,
                baseResult.getFeatureSchema(),
                baseBorders,
                baseResult.getBias(),
                trainingRun.losses
        );
    }

    private TrainingRunResult train(Dataset dataset,
                                    QuantizedDataset quantizedDataset,
                                    OrderedTrainingState trainingState,
                                    double bias) {
        TrainingDebugLog.log("fit start rows=%d features=%d bias=%.12f seed=%d", dataset.getRowCount(), dataset.getFeatureSchema().size(), bias, config.getRandomSeed());
        for (int featureIndex = 0; featureIndex < quantizedDataset.getFeatureCount(); featureIndex++) {
            TrainingDebugLog.log(
                    "feature=%d borders=%s",
                    featureIndex,
                    TrainingDebugLog.formatDoubleArray(quantizedDataset.getBorders(featureIndex))
            );
        }
        int[] rowLeafIndexes = new int[dataset.getRowCount()];
        List<ObliviousTree> trees = new ArrayList<ObliviousTree>();
        List<Double> losses = new ArrayList<Double>();
        losses.add(trainingState.computeLoss(lossFunction));
        TrainingDebugLog.log("initial loss=%.12f", losses.get(0));

        for (int iteration = 0; iteration < config.getIterations(); iteration++) {
            IterationContext iterationContext = trainingState.beginIteration(config);
            TrainingDebugLog.log(
                    "iteration=%d selectedFold=%d scoreStdDev=%.12f bootstrap=%s",
                    iteration,
                    iterationContext.getSelectedLearnFoldIndex(),
                    iterationContext.getScoreStdDev(),
                    TrainingDebugLog.formatDoubleArray(iterationContext.getBootstrapWeights())
            );
            ObliviousTreeBuilder.TreeBuildResult buildResult = treeBuilder.buildOrdered(
                    quantizedDataset,
                    trainingState,
                    iterationContext,
                    config,
                    rowLeafIndexes
            );
            trainingState.getRandom().advance(1);
            trainingState.finishIteration();
            ObliviousTree tree = buildResult.getTree();
            trees.add(tree);
            trainingState.applyTree(quantizedDataset, buildResult);
            losses.add(trainingState.computeLoss(lossFunction));
            TrainingDebugLog.log(
                    "iteration=%d treeSplits=%s leafValues=%s loss=%.12f",
                    iteration,
                    TrainingDebugLog.formatSplits(tree.getSplits()),
                    TrainingDebugLog.formatDoubleArray(tree.getLeafValues()),
                    losses.get(losses.size() - 1)
            );
        }
        return new TrainingRunResult(trees, losses);
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

    private CompactedModel compactModel(List<ObliviousTree> trees, QuantizedDataset quantizedDataset) {
        double[][] sourceBorders = extractBorders(quantizedDataset);
        int featureCount = quantizedDataset.getFeatureCount();
        boolean[][] usedBorders = new boolean[featureCount][];
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            usedBorders[featureIndex] = new boolean[sourceBorders[featureIndex].length];
        }
        for (ObliviousTree tree : trees) {
            for (ObliviousSplit split : tree.getSplits()) {
                usedBorders[split.getFeatureIndex()][split.getBorderIndex()] = true;
            }
        }

        double[][] compactBorders = new double[featureCount][];
        int[][] borderRemap = new int[featureCount][];
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            int[] remap = new int[sourceBorders[featureIndex].length];
            Arrays.fill(remap, -1);
            int usedCount = 0;
            for (int borderIndex = 0; borderIndex < usedBorders[featureIndex].length; borderIndex++) {
                if (usedBorders[featureIndex][borderIndex]) {
                    remap[borderIndex] = usedCount++;
                }
            }

            compactBorders[featureIndex] = new double[usedCount];
            for (int borderIndex = 0; borderIndex < usedBorders[featureIndex].length; borderIndex++) {
                if (usedBorders[featureIndex][borderIndex]) {
                    compactBorders[featureIndex][remap[borderIndex]] = sourceBorders[featureIndex][borderIndex];
                }
            }
            borderRemap[featureIndex] = remap;
        }

        List<ObliviousTree> compactTrees = new ArrayList<ObliviousTree>(trees.size());
        for (ObliviousTree tree : trees) {
            List<ObliviousSplit> compactSplits = new ArrayList<ObliviousSplit>(tree.getSplits().size());
            for (ObliviousSplit split : tree.getSplits()) {
                compactSplits.add(new ObliviousSplit(
                        split.getFeatureIndex(),
                        borderRemap[split.getFeatureIndex()][split.getBorderIndex()]
                ));
            }
            compactTrees.add(new ObliviousTree(compactSplits, tree.getLeafValues()));
        }
        return new CompactedModel(compactTrees, compactBorders);
    }

    private void validateWarmStartCompatibility(Dataset dataset, TrainingResult baseResult) {
        if (dataset.getRowCount() == 0) {
            throw new IllegalArgumentException("warm-start dataset must contain at least one row");
        }
        if (dataset.getFeatureCount() != baseResult.getFeatureSchema().size()) {
            throw new IllegalArgumentException("warm-start feature count must match base result feature count");
        }
        for (int featureIndex = 0; featureIndex < dataset.getFeatureCount(); featureIndex++) {
            String expected = baseResult.getFeatureSchema().getFeatureName(featureIndex);
            String actual = dataset.getFeatureSchema().getFeatureName(featureIndex);
            if (!expected.equals(actual)) {
                throw new IllegalArgumentException(
                        "warm-start feature name mismatch at index " + featureIndex + ": expected " + expected + " but was " + actual
                );
            }
        }
        double[][] borders = baseResult.getBorders();
        if (borders.length != dataset.getFeatureCount()) {
            throw new IllegalArgumentException("base borders feature count must match dataset feature count");
        }
    }

    private static final class CompactedModel {
        private final List<ObliviousTree> trees;
        private final double[][] borders;

        private CompactedModel(List<ObliviousTree> trees, double[][] borders) {
            this.trees = trees;
            this.borders = borders;
        }
    }

    private static final class TrainingRunResult {
        private final List<ObliviousTree> trees;
        private final List<Double> losses;

        private TrainingRunResult(List<ObliviousTree> trees, List<Double> losses) {
            this.trees = trees;
            this.losses = losses;
        }
    }
}
