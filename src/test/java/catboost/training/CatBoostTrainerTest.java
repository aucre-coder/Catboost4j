package catboost.training;

import catboost.model.Model;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CatBoostTrainerTest {

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
                .setL2LeafReg(1.0);
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
}
