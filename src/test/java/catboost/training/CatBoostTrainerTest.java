package catboost.training;

import catboost.model.Model;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CatBoostTrainerTest {

    private static final Gson GSON = new Gson();
    private static final String WINE_QUALITY_RED_URL =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv";
    private static final String WINE_QUALITY_WHITE_URL =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv";

    @Test
    void trainingLossDecreasesAndModelPredictsUsefulValues() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 0.0},
                        {1.0, 1.0},
                        {2.0, 1.0},
                        {2.0, 2.0}
                },
                new double[]{0.0, 3.0, 2.0, 5.0, 7.0, 10.0},
                Arrays.asList("x1", "x2")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(25)
                .setDepth(2)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0);
        CatBoostTrainer trainer = new CatBoostTrainer(config);

        TrainingResult result = trainer.fitResult(dataset);
        Model model = trainer.fit(dataset);

        assertEquals(config.getIterations() + 1, result.getTrainingLosses().size());
        assertTrue(result.getTrainingLosses().get(result.getTrainingLosses().size() - 1) < result.getTrainingLosses().get(0));
        assertTrue(model.predict(features(0.0, 0.0)) < model.predict(features(2.0, 2.0)));
        assertTrue(result.getTrainingLosses().get(result.getTrainingLosses().size() - 1) < 2.5);
        assertTrue(Math.abs(model.predict(features(0.0, 0.0)) - 0.0) < 3.0);
        assertTrue(Math.abs(model.predict(features(2.0, 2.0)) - 10.0) < 5.0);
    }

    @Test
    void matchesFixedOrderedSingleTreeParityCase() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0},
                        {2.0},
                        {3.0}
                },
                new double[]{0.0, 0.0, 2.0, 2.0},
                Arrays.asList("x1")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(1)
                .setDepth(1)
                .setLearningRate(1.0)
                .setMaxBins(2)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        CatBoostTrainer trainer = new CatBoostTrainer(config);
        TrainingResult result = trainer.fitResult(dataset);
        Model model = trainer.fit(dataset);

        assertArrayEquals(new double[]{1.5}, result.getBorders()[0], 1e-12);
        assertEquals(1, result.getTrees().size());
        ObliviousTree tree = result.getTrees().get(0);
        assertEquals(1, tree.getSplits().size());
        assertEquals(0, tree.getSplits().get(0).getFeatureIndex());
        assertEquals(0, tree.getSplits().get(0).getBorderIndex());
        assertArrayEquals(new double[]{-2.0 / 3.0, 2.0 / 3.0}, tree.getLeafValues(), 1e-12);
        assertEquals(1.0 / 3.0, model.predict(feature("x1", 0.0)), 1e-12);
        assertEquals(1.0 / 3.0, model.predict(feature("x1", 1.0)), 1e-12);
        assertEquals(5.0 / 3.0, model.predict(feature("x1", 2.0)), 1e-12);
        assertEquals(5.0 / 3.0, model.predict(feature("x1", 3.0)), 1e-12);
    }

    @Test
    void binaryRegressionPredictionsCrossFiftyPercentWhenReadDirectly() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {0.0},
                        {1.0},
                        {1.0}
                },
                new double[]{0.0, 0.0, 1.0, 1.0},
                Arrays.asList("x1")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(20)
                .setDepth(1)
                .setLearningRate(0.5)
                .setMaxBins(2)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        Model model = new CatBoostTrainer(config).fit(dataset);

        assertTrue(model.predict(feature("x1", 0.0)) < 0.5);
        assertTrue(model.predict(feature("x1", 1.0)) > 0.5);
        assertTrue(model.predictBoundedProbability(feature("x1", 0.0)) < 0.5);
        assertTrue(model.predictBoundedProbability(feature("x1", 1.0)) > 0.5);
    }

    @Test
    void matchesCatBoostOrderedMultiIterationParityCase() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 0.0},
                        {1.0, 1.0},
                        {2.0, 1.0},
                        {2.0, 2.0}
                },
                new double[]{0.0, 3.0, 2.0, 5.0, 7.0, 10.0},
                Arrays.asList("x1", "x2")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(3)
                .setDepth(2)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        CatBoostTrainer trainer = new CatBoostTrainer(config);
        TrainingResult result = trainer.fitResult(dataset);
        Model model = trainer.fit(dataset);

        assertArrayEquals(new double[]{1.5}, result.getBorders()[0], 1e-12);
        assertArrayEquals(new double[]{0.5}, result.getBorders()[1], 1e-12);
        assertEquals(3, result.getTrees().size());

        assertTreeMatches(result.getTrees().get(0), new int[]{0, 1}, new int[]{0, 0},
                new double[]{-0.7000000278155009, 0.0, -0.10000000397364298, 0.8000000317891438});
        assertTreeMatches(result.getTrees().get(1), new int[]{0, 1}, new int[]{0, 0},
                new double[]{-0.5600000166893003, 0.0, -0.08000000238418568, 0.640000019073486});
        assertTreeMatches(result.getTrees().get(2), new int[]{0, 1}, new int[]{0, 0},
                new double[]{-0.44800000890096, 0.0, -0.06400000127156573, 0.5120000101725258});

        assertEquals(2.791999946594239, model.predict(features(0.0, 0.0)), 1e-6);
        assertEquals(4.255999992370605, model.predict(features(0.0, 1.0)), 1e-6);
        assertEquals(2.791999946594239, model.predict(features(1.0, 0.0)), 1e-6);
        assertEquals(4.255999992370605, model.predict(features(1.0, 1.0)), 1e-6);
        assertEquals(6.4520000610351556, model.predict(features(2.0, 1.0)), 1e-6);
        assertEquals(6.4520000610351556, model.predict(features(2.0, 2.0)), 1e-6);
    }

    @Test
    void matchesCatBoostOrderedSingleFeatureTwoLevelFixture() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0},
                        {2.0},
                        {3.0},
                        {4.0},
                        {5.0}
                },
                new double[]{0.0, 1.0, 1.5, 3.0, 3.5, 5.0},
                Arrays.asList("x1")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(1)
                .setDepth(2)
                .setLearningRate(0.4)
                .setMaxBins(5)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        CatBoostTrainer trainer = new CatBoostTrainer(config);
        TrainingResult result = trainer.fitResult(dataset);
        Model model = trainer.fit(dataset);

        assertArrayEquals(new double[]{2.5, 3.5}, result.getBorders()[0], 1e-12);
        assertEquals(1, result.getTrees().size());

        assertTreeMatches(
                result.getTrees().get(0),
                new int[]{0, 0},
                new int[]{0, 1},
                new double[]{-0.44999998286366427, 0.133333351214727, 0.0, 0.5111111399200231}
        );

        assertEquals(1.8833332709968094, model.predict(feature("x1", 0.0)), 1e-6);
        assertEquals(1.8833332709968094, model.predict(feature("x1", 1.0)), 1e-6);
        assertEquals(1.8833332709968094, model.predict(feature("x1", 2.0)), 1e-6);
        assertEquals(2.4666666050752006, model.predict(feature("x1", 3.0)), 1e-6);
        assertEquals(2.844444393780497, model.predict(feature("x1", 4.0)), 1e-6);
        assertEquals(2.844444393780497, model.predict(feature("x1", 5.0)), 1e-6);
    }

    @Test
    void stochasticTrainingIsSeededAndRepeatable() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0},
                        {2.0},
                        {3.0},
                        {4.0},
                        {5.0}
                },
                new double[]{0.0, 1.0, 1.5, 3.0, 3.5, 5.0},
                Arrays.asList("x1")
        );

        TrainerConfig seededConfig = new TrainerConfig()
                .setIterations(5)
                .setDepth(2)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.BAYESIAN)
                .setBaggingTemperature(1.0)
                .setRandomStrength(1.0)
                .setRandomSeed(7L);
        TrainerConfig otherSeedConfig = new TrainerConfig()
                .setIterations(5)
                .setDepth(2)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.BAYESIAN)
                .setBaggingTemperature(1.0)
                .setRandomStrength(1.0)
                .setRandomSeed(8L);

        TrainingResult first = new CatBoostTrainer(seededConfig).fitResult(dataset);
        TrainingResult second = new CatBoostTrainer(seededConfig).fitResult(dataset);
        TrainingResult third = new CatBoostTrainer(otherSeedConfig).fitResult(dataset);

        assertTreeMatches(
                first.getTrees().get(0),
                treeFeatures(second.getTrees().get(0)),
                treeBorders(second.getTrees().get(0)),
                second.getTrees().get(0).getLeafValues()
        );
        assertArrayEquals(first.getTrees().get(0).getLeafValues(), second.getTrees().get(0).getLeafValues(), 1e-12);
        assertTrue(
                !Arrays.equals(first.getTrees().get(0).getLeafValues(), third.getTrees().get(0).getLeafValues())
                        || treeFeatures(first.getTrees().get(0))[0] != treeFeatures(third.getTrees().get(0))[0]
                        || treeBorders(first.getTrees().get(0))[0] != treeBorders(third.getTrees().get(0))[0]
        );
    }

    @Test
    void orderedTrainingWithBootstrapNoAndRandomStrengthZeroIsExactlyDeterministic() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 0.0},
                        {1.0, 1.0},
                        {2.0, 1.0},
                        {2.0, 2.0}
                },
                new double[]{0.0, 3.0, 2.0, 5.0, 7.0, 10.0},
                Arrays.asList("x1", "x2")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(4)
                .setDepth(2)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(123L);

        TrainingResult first = new CatBoostTrainer(config).fitResult(dataset);
        TrainingResult second = new CatBoostTrainer(config).fitResult(dataset);

        assertArrayEquals(first.getBorders()[0], second.getBorders()[0], 0.0);
        assertArrayEquals(first.getBorders()[1], second.getBorders()[1], 0.0);
        assertEquals(first.getTrainingLosses(), second.getTrainingLosses());
        assertEquals(first.getTrees().size(), second.getTrees().size());
        for (int treeIndex = 0; treeIndex < first.getTrees().size(); treeIndex++) {
            assertTreeMatches(
                    first.getTrees().get(treeIndex),
                    treeFeatures(second.getTrees().get(treeIndex)),
                    treeBorders(second.getTrees().get(treeIndex)),
                    second.getTrees().get(treeIndex).getLeafValues()
            );
        }
    }

    @Test
    void matchesWeightedDeterministicSingleTreeFixture() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0},
                        {2.0},
                        {3.0}
                },
                new double[]{0.0, 0.0, 2.0, 2.0},
                new double[]{1.0, 1.0, 2.0, 2.0},
                Arrays.asList("x1")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(1)
                .setDepth(1)
                .setLearningRate(1.0)
                .setMaxBins(2)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        TrainingResult result = new CatBoostTrainer(config).fitResult(dataset);

        assertArrayEquals(new double[]{1.5}, result.getBorders()[0], 1e-12);
        assertTreeMatches(
                result.getTrees().get(0),
                new int[]{0},
                new int[]{0},
                new double[]{-0.7619047619047619, 0.48484848484848486}
        );
    }

    @Test
    void rejectsBootstrapModesWithoutStrictParityCoverage() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0}
                },
                new double[]{0.0, 1.0},
                Arrays.asList("x1")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(1)
                .setDepth(1)
                .setBootstrapType(BootstrapType.BERNOULLI);

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> new CatBoostTrainer(config).fitResult(dataset)
        );

        assertTrue(error.getMessage().contains("strict native-Java CatBoost parity scope"));
    }

    @Test
    void comparesPerformanceAgainstPythonCatBoostForOrderedRmseSymmetricTree() throws Exception {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 0.0},
                        {1.0, 1.0},
                        {2.0, 1.0},
                        {2.0, 2.0},
                        {3.0, 1.0},
                        {3.0, 2.0}
                },
                new double[]{0.0, 3.0, 2.0, 5.0, 7.0, 10.0, 11.0, 13.0},
                new double[]{1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0},
                Arrays.asList("x1", "x2")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(5)
                .setDepth(2)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        PythonComparison pythonComparison = runPythonCatBoost(dataset, config);
        assumeTrue(pythonComparison != null, "Python CatBoost is unavailable in this environment");

        TrainingResult javaResult = new CatBoostTrainer(config).fitResult(dataset);
        Model javaModel = new CatBoostTrainer(config).fit(dataset);
        double[] javaPredictions = predictDataset(javaModel, dataset);
        double javaRmse = weightedRmse(javaPredictions, dataset.getTargets(), dataset.getWeights());

        assertEquals(pythonComparison.rmse, javaRmse, 2e-2);
        assertArrayEquals(pythonComparison.predictions, javaPredictions, 3e-2);

        assertTrue(
                Math.abs(pythonComparison.rmse - javaResult.getTrainingLosses().get(javaResult.getTrainingLosses().size() - 1)) <= 2e-2
        );
    }

    @Test
    void comparesAgainstPythonCatBoostOnDownloadedWineQualityDataset() throws Exception {
        RegressionSplit split = loadWineQualityRedSplit();
        TrainerConfig config = new TrainerConfig()
                .setIterations(30)
                .setDepth(4)
                .setLearningRate(0.1)
                .setMaxBins(32)
                .setL2LeafReg(3.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        PythonComparison pythonComparison = runPythonCatBoost(split.training, split.evaluation, config);
        assumeTrue(pythonComparison != null, "Python CatBoost is unavailable in this environment");

        Model javaModel = new CatBoostTrainer(config).fit(split.training);
        double[] javaPredictions = predictDataset(javaModel, split.evaluation);
        double javaRmse = weightedRmse(javaPredictions, split.evaluation.getTargets(), split.evaluation.getWeights());

        assertTrue(Math.abs(pythonComparison.rmse - javaRmse) <= 0.2,
                "expected downloaded-data RMSE gap <= 0.2 but was "
                        + Math.abs(pythonComparison.rmse - javaRmse)
                        + " (python=" + pythonComparison.rmse + ", java=" + javaRmse + ")");
        assertArrayEquals(pythonComparison.predictions, javaPredictions, 0.7);
    }

    @Test
    void comparesAgainstPythonCatBoostOnDownloadedCombinedWineQualityDataset() throws Exception {
        RegressionSplit split = loadCombinedWineQualitySplit();
        TrainerConfig config = new TrainerConfig()
                .setIterations(40)
                .setDepth(5)
                .setLearningRate(0.1)
                .setMaxBins(32)
                .setL2LeafReg(3.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        PythonComparison pythonComparison = runPythonCatBoost(split.training, split.evaluation, config);
        assumeTrue(pythonComparison != null, "Python CatBoost is unavailable in this environment");

        Model javaModel = new CatBoostTrainer(config).fit(split.training);
        double[] javaPredictions = predictDataset(javaModel, split.evaluation);
        double javaRmse = weightedRmse(javaPredictions, split.evaluation.getTargets(), split.evaluation.getWeights());

        assertTrue(Math.abs(pythonComparison.rmse - javaRmse) <= 0.2,
                "expected combined downloaded-data RMSE gap <= 0.2 but was "
                        + Math.abs(pythonComparison.rmse - javaRmse)
                        + " (python=" + pythonComparison.rmse + ", java=" + javaRmse + ")");
        assertArrayEquals(pythonComparison.predictions, javaPredictions, 0.8);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("combinedWineQualityComparisonConfigs")
    void comparesAgainstPythonCatBoostAcrossCombinedWineQualityParameterSweep(String caseName,
                                                                              int iterations,
                                                                              int depth,
                                                                              double learningRate,
                                                                              int maxBins,
                                                                              double l2LeafReg,
                                                                              double maxRmseGap,
                                                                              double maxPredictionGap) throws Exception {
        RegressionSplit split = loadCombinedWineQualitySplit();
        TrainerConfig config = new TrainerConfig()
                .setIterations(iterations)
                .setDepth(depth)
                .setLearningRate(learningRate)
                .setMaxBins(maxBins)
                .setL2LeafReg(l2LeafReg)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        PythonComparison pythonComparison = runPythonCatBoost(split.training, split.evaluation, config);
        assumeTrue(pythonComparison != null, "Python CatBoost is unavailable in this environment");

        Model javaModel = new CatBoostTrainer(config).fit(split.training);
        double[] javaPredictions = predictDataset(javaModel, split.evaluation);
        double javaRmse = weightedRmse(javaPredictions, split.evaluation.getTargets(), split.evaluation.getWeights());

        assertTrue(Math.abs(pythonComparison.rmse - javaRmse) <= maxRmseGap,
                caseName + " expected RMSE gap <= " + maxRmseGap + " but was "
                        + Math.abs(pythonComparison.rmse - javaRmse)
                        + " (python=" + pythonComparison.rmse + ", java=" + javaRmse + ")");
        assertArrayEquals(pythonComparison.predictions, javaPredictions, maxPredictionGap);
    }

    @Test
    void doesNotReuseTheSameSplitTwiceInsideOneSymmetricTree() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0},
                        {2.0},
                        {3.0},
                        {4.0},
                        {5.0},
                        {6.0},
                        {7.0}
                },
                new double[]{0.0, 0.0, 1.0, 1.5, 4.0, 4.5, 7.0, 7.5},
                Arrays.asList("x1")
        );
        TrainerConfig config = new TrainerConfig()
                .setIterations(1)
                .setDepth(3)
                .setLearningRate(0.3)
                .setMaxBins(8)
                .setL2LeafReg(1.0)
                .setBootstrapType(BootstrapType.NO)
                .setRandomStrength(0.0)
                .setRandomSeed(0L);

        TrainingResult result = new CatBoostTrainer(config).fitResult(dataset);
        ObliviousTree tree = result.getTrees().get(0);
        Set<String> seenSplits = new HashSet<String>();
        for (ObliviousSplit split : tree.getSplits()) {
            String splitKey = split.getFeatureIndex() + ":" + split.getBorderIndex();
            assertTrue(seenSplits.add(splitKey), "duplicate split reused inside one symmetric tree: " + splitKey);
        }
    }

    private void assertTreeMatches(ObliviousTree tree,
                                   int[] expectedFeatures,
                                   int[] expectedBorders,
                                   double[] expectedLeafValues) {
        assertEquals(expectedFeatures.length, tree.getSplits().size());
        for (int splitIndex = 0; splitIndex < expectedFeatures.length; splitIndex++) {
            assertEquals(expectedFeatures[splitIndex], tree.getSplits().get(splitIndex).getFeatureIndex());
            assertEquals(expectedBorders[splitIndex], tree.getSplits().get(splitIndex).getBorderIndex());
        }
        assertArrayEquals(expectedLeafValues, tree.getLeafValues(), 1e-6);
    }

    private Map<String, String> features(double x1, double x2) {
        Map<String, String> input = new HashMap<String, String>();
        input.put("x1", String.valueOf(x1));
        input.put("x2", String.valueOf(x2));
        return input;
    }

    private Map<String, String> feature(String name, double value) {
        Map<String, String> input = new HashMap<String, String>();
        input.put(name, String.valueOf(value));
        return input;
    }

    private int[] treeFeatures(ObliviousTree tree) {
        int[] features = new int[tree.getSplits().size()];
        for (int i = 0; i < tree.getSplits().size(); i++) {
            features[i] = tree.getSplits().get(i).getFeatureIndex();
        }
        return features;
    }

    private int[] treeBorders(ObliviousTree tree) {
        int[] borders = new int[tree.getSplits().size()];
        for (int i = 0; i < tree.getSplits().size(); i++) {
            borders[i] = tree.getSplits().get(i).getBorderIndex();
        }
        return borders;
    }

    private double[] predictDataset(Model model, Dataset dataset) {
        double[][] features = dataset.getFloatFeatures();
        FeatureSchema schema = dataset.getFeatureSchema();
        double[] predictions = new double[features.length];
        for (int row = 0; row < features.length; row++) {
            Map<String, String> input = new HashMap<String, String>();
            for (int featureIndex = 0; featureIndex < features[row].length; featureIndex++) {
                input.put(schema.getFeatureName(featureIndex), String.valueOf(features[row][featureIndex]));
            }
            predictions[row] = model.predict(input);
        }
        return predictions;
    }

    private double weightedRmse(double[] predictions, double[] targets, double[] weights) {
        double weightedSquaredError = 0.0;
        double totalWeight = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double weight = weights == null ? 1.0 : weights[i];
            double diff = predictions[i] - targets[i];
            weightedSquaredError += diff * diff * weight;
            totalWeight += weight;
        }
        return Math.sqrt(weightedSquaredError / totalWeight);
    }

    private PythonComparison runPythonCatBoost(Dataset dataset, TrainerConfig config) throws Exception {
        return runPythonCatBoost(dataset, dataset, config);
    }

    private PythonComparison runPythonCatBoost(Dataset trainingDataset, Dataset evaluationDataset, TrainerConfig config) throws Exception {
        if (!commandSucceeds("python", "--version") || !commandSucceeds("python", "-c", "import catboost")) {
            return null;
        }

        Path requestFile = Files.createTempFile("catboost-python-compare-", ".json");
        Path responseFile = Files.createTempFile("catboost-python-response-", ".json");
        try {
            Files.write(
                    requestFile,
                    GSON.toJson(buildPythonRequest(trainingDataset, evaluationDataset, config)).getBytes(StandardCharsets.UTF_8)
            );
            Process process = new ProcessBuilder(
                    "python",
                    "-c",
                    pythonComparisonScript(),
                    requestFile.toString(),
                    responseFile.toString()
            ).start();
            int exitCode = process.waitFor();
            String stderr = readProcessStream(process.getErrorStream());
            assumeTrue(exitCode == 0, "Python CatBoost comparison failed: " + stderr);

            Type responseType = new TypeToken<Map<String, Object>>() { }.getType();
            Map<String, Object> response = GSON.fromJson(
                    new String(Files.readAllBytes(responseFile), StandardCharsets.UTF_8),
                    responseType
            );
            List<Double> predictionsList = castDoubleList(response.get("predictions"));
            return new PythonComparison(
                    toDoubleArray(predictionsList),
                    ((Number) response.get("rmse")).doubleValue()
            );
        } finally {
            Files.deleteIfExists(requestFile);
            Files.deleteIfExists(responseFile);
        }
    }

    private Map<String, Object> buildPythonRequest(Dataset trainingDataset, Dataset evaluationDataset, TrainerConfig config) {
        Map<String, Object> request = new LinkedHashMap<String, Object>();
        request.put("train_features", trainingDataset.getFloatFeatures());
        request.put("train_targets", trainingDataset.getTargets());
        request.put("train_weights", trainingDataset.getWeights());
        request.put("predict_features", evaluationDataset.getFloatFeatures());
        request.put("predict_targets", evaluationDataset.getTargets());
        request.put("predict_weights", evaluationDataset.getWeights());
        request.put("iterations", config.getIterations());
        request.put("depth", config.getDepth());
        request.put("learning_rate", config.getLearningRate());
        request.put("l2_leaf_reg", config.getL2LeafReg());
        request.put("border_count", config.getMaxBins() - 1);
        request.put("random_strength", config.getRandomStrength());
        request.put("random_seed", config.getRandomSeed());
        request.put("bootstrap_type", config.getBootstrapType().name());
        if (config.getBootstrapType() == BootstrapType.BAYESIAN) {
            request.put("bagging_temperature", config.getBaggingTemperature());
        }
        return request;
    }

    private String pythonComparisonScript() {
        return ""
                + "import json, math, sys\n"
                + "from catboost import CatBoostRegressor, Pool\n"
                + "request_path, response_path = sys.argv[1], sys.argv[2]\n"
                + "with open(request_path, 'r', encoding='utf-8') as fh:\n"
                + "    req = json.load(fh)\n"
                + "pool = Pool(req['train_features'], label=req['train_targets'], weight=req.get('train_weights'))\n"
                + "model = CatBoostRegressor(\n"
                + "    loss_function='RMSE',\n"
                + "    task_type='CPU',\n"
                + "    boosting_type='Ordered',\n"
                + "    grow_policy='SymmetricTree',\n"
                + "    leaf_estimation_method='Gradient',\n"
                + "    leaf_estimation_iterations=1,\n"
                + "    score_function='Cosine',\n"
                + "    feature_border_type='GreedyLogSum',\n"
                + "    iterations=req['iterations'],\n"
                + "    depth=req['depth'],\n"
                + "    learning_rate=req['learning_rate'],\n"
                + "    l2_leaf_reg=req['l2_leaf_reg'],\n"
                + "    border_count=req['border_count'],\n"
                + "    random_strength=req['random_strength'],\n"
                + "    random_seed=req['random_seed'],\n"
                + "    bootstrap_type=req['bootstrap_type'].title(),\n"
                + "    verbose=False,\n"
                + "    allow_writing_files=False)\n"
                + "if 'bagging_temperature' in req:\n"
                + "    model.set_params(bagging_temperature=req['bagging_temperature'])\n"
                + "model.fit(pool)\n"
                + "preds = model.predict(req['predict_features'])\n"
                + "weighted_sq = 0.0\n"
                + "total_weight = 0.0\n"
                + "weights = req.get('predict_weights') if req.get('predict_weights') is not None else [1.0] * len(preds)\n"
                + "for pred, target, weight in zip(preds, req['predict_targets'], weights):\n"
                + "    diff = pred - target\n"
                + "    weighted_sq += diff * diff * weight\n"
                + "    total_weight += weight\n"
                + "rmse = math.sqrt(weighted_sq / total_weight)\n"
                + "with open(response_path, 'w', encoding='utf-8') as fh:\n"
                + "    json.dump({'predictions': list(map(float, preds)), 'rmse': rmse}, fh)\n";
    }

    private RegressionSplit loadWineQualityRedSplit() throws IOException {
        Path datasetPath = downloadIfMissing(
                WINE_QUALITY_RED_URL,
                Path.of("target", "downloaded-test-data", "winequality-red.csv")
        );
        return parseWineQualityRedSplit(datasetPath, 0.8);
    }

    private RegressionSplit loadCombinedWineQualitySplit() throws IOException {
        Path redPath = downloadIfMissing(
                WINE_QUALITY_RED_URL,
                Path.of("target", "downloaded-test-data", "winequality-red.csv")
        );
        Path whitePath = downloadIfMissing(
                WINE_QUALITY_WHITE_URL,
                Path.of("target", "downloaded-test-data", "winequality-white.csv")
        );
        return parseCombinedWineQualitySplit(redPath, whitePath, 0.8);
    }

    private Path downloadIfMissing(String url, Path path) throws IOException {
        if (Files.exists(path) && Files.size(path) > 0) {
            return path;
        }

        Files.createDirectories(path.getParent());
        try (java.io.InputStream stream = new URL(url).openStream()) {
            Files.copy(stream, path, StandardCopyOption.REPLACE_EXISTING);
        }
        return path;
    }

    private RegressionSplit parseWineQualityRedSplit(Path datasetPath, double trainingFraction) throws IOException {
        List<String> lines = Files.readAllLines(datasetPath, StandardCharsets.UTF_8);
        if (lines.size() < 3) {
            throw new IllegalArgumentException("dataset must contain a header and at least two rows");
        }

        String[] header = splitSemicolonCsvLine(lines.get(0));
        int targetIndex = header.length - 1;
        List<String> featureNames = new ArrayList<String>(targetIndex);
        for (int i = 0; i < targetIndex; i++) {
            featureNames.add(stripCsvValue(header[i]));
        }

        int rowCount = lines.size() - 1;
        int trainingRows = Math.max(1, Math.min(rowCount - 1, (int) Math.round(rowCount * trainingFraction)));
        int evaluationRows = rowCount - trainingRows;

        double[][] trainingFeatures = new double[trainingRows][targetIndex];
        double[] trainingTargets = new double[trainingRows];
        double[][] evaluationFeatures = new double[evaluationRows][targetIndex];
        double[] evaluationTargets = new double[evaluationRows];

        for (int row = 0; row < rowCount; row++) {
            String[] values = splitSemicolonCsvLine(lines.get(row + 1));
            double[] features = new double[targetIndex];
            for (int featureIndex = 0; featureIndex < targetIndex; featureIndex++) {
                features[featureIndex] = Double.parseDouble(stripCsvValue(values[featureIndex]));
            }
            double target = Double.parseDouble(stripCsvValue(values[targetIndex]));
            if (row < trainingRows) {
                trainingFeatures[row] = features;
                trainingTargets[row] = target;
            } else {
                evaluationFeatures[row - trainingRows] = features;
                evaluationTargets[row - trainingRows] = target;
            }
        }

        return new RegressionSplit(
                Dataset.of(trainingFeatures, trainingTargets, featureNames),
                Dataset.of(evaluationFeatures, evaluationTargets, featureNames)
        );
    }

    private RegressionSplit parseCombinedWineQualitySplit(Path redDatasetPath,
                                                          Path whiteDatasetPath,
                                                          double trainingFraction) throws IOException {
        List<String> combinedLines = new ArrayList<String>();
        combinedLines.addAll(Files.readAllLines(redDatasetPath, StandardCharsets.UTF_8));
        List<String> whiteLines = Files.readAllLines(whiteDatasetPath, StandardCharsets.UTF_8);
        for (int i = 1; i < whiteLines.size(); i++) {
            combinedLines.add(whiteLines.get(i));
        }

        Path mergedDatasetPath = Files.createTempFile("winequality-combined-", ".csv");
        try {
            Files.write(mergedDatasetPath, combinedLines, StandardCharsets.UTF_8);
            return parseWineQualityRedSplit(mergedDatasetPath, trainingFraction);
        } finally {
            Files.deleteIfExists(mergedDatasetPath);
        }
    }

    private static Stream<Object[]> combinedWineQualityComparisonConfigs() {
        return Stream.of(
                new Object[]{"combined-depth4-it30-lr0.10-b32-l2-3", 30, 4, 0.10, 32, 3.0, 0.02, 0.35},
                new Object[]{"combined-depth5-it40-lr0.10-b32-l2-3", 40, 5, 0.10, 32, 3.0, 0.02, 0.30},
                new Object[]{"combined-depth6-it60-lr0.08-b32-l2-3", 60, 6, 0.08, 32, 3.0, 0.02, 0.40},
                new Object[]{"combined-depth3-it80-lr0.05-b64-l2-5", 80, 3, 0.05, 64, 5.0, 0.02, 0.30},
                new Object[]{"combined-depth6-it100-lr0.03-b64-l2-10", 100, 6, 0.03, 64, 10.0, 0.02, 0.30}
        );
    }

    private String[] splitSemicolonCsvLine(String line) {
        return line.split(";");
    }

    private String stripCsvValue(String value) {
        String trimmed = value.trim();
        if (trimmed.startsWith("\"") && trimmed.endsWith("\"") && trimmed.length() >= 2) {
            return trimmed.substring(1, trimmed.length() - 1);
        }
        return trimmed;
    }

    private boolean commandSucceeds(String... command) throws IOException, InterruptedException {
        Process process = new ProcessBuilder(command).start();
        int exitCode = process.waitFor();
        return exitCode == 0;
    }

    private String readProcessStream(java.io.InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
        StringBuilder builder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            if (builder.length() > 0) {
                builder.append('\n');
            }
            builder.append(line);
        }
        return builder.toString();
    }

    @SuppressWarnings("unchecked")
    private List<Double> castDoubleList(Object value) {
        return (List<Double>) value;
    }

    private double[] toDoubleArray(List<Double> values) {
        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i);
        }
        return result;
    }

    private static final class PythonComparison {
        private final double[] predictions;
        private final double rmse;

        private PythonComparison(double[] predictions, double rmse) {
            this.predictions = predictions;
            this.rmse = rmse;
        }
    }

    private static final class RegressionSplit {
        private final Dataset training;
        private final Dataset evaluation;

        private RegressionSplit(Dataset training, Dataset evaluation) {
            this.training = training;
            this.evaluation = evaluation;
        }
    }
}
