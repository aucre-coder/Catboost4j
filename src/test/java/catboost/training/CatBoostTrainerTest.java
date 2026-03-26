package catboost.training;

import catboost.model.Model;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

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
        assertTrue(result.getTrainingLosses().get(result.getTrainingLosses().size() - 1) < 0.5);
        assertTrue(Math.abs(model.predict(features(0.0, 0.0)) - 0.0) < 1.0);
        assertTrue(Math.abs(model.predict(features(2.0, 2.0)) - 10.0) < 1.0);
    }

    private Map<String, String> features(double x1, double x2) {
        Map<String, String> input = new HashMap<String, String>();
        input.put("x1", String.valueOf(x1));
        input.put("x2", String.valueOf(x2));
        return input;
    }
}
