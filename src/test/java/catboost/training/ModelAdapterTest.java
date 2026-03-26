package catboost.training;

import catboost.model.Model;
import catboost.training.adapter.ModelAdapter;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ModelAdapterTest {

    @Test
    void convertsLeafOrderingIntoPredictorTreeNodes() {
        ObliviousTree tree = new ObliviousTree(
                Arrays.asList(
                        new ObliviousSplit(0, 0),
                        new ObliviousSplit(1, 0)
                ),
                new double[]{10.0, 20.0, 30.0, 40.0}
        );
        TrainingResult result = new TrainingResult(
                Collections.singletonList(tree),
                new FeatureSchema(Arrays.asList("x1", "x2")),
                new double[][]{
                        {0.5},
                        {1.5}
                },
                2.0,
                Collections.singletonList(0.0)
        );

        Model model = new ModelAdapter().toModel(result);

        assertEquals(12.0, model.predict(features(0.0, 1.0)), 1e-12);
        assertEquals(22.0, model.predict(features(1.0, 1.0)), 1e-12);
        assertEquals(32.0, model.predict(features(0.0, 2.0)), 1e-12);
        assertEquals(42.0, model.predict(features(1.0, 2.0)), 1e-12);
    }

    private Map<String, String> features(double x1, double x2) {
        Map<String, String> input = new HashMap<String, String>();
        input.put("x1", String.valueOf(x1));
        input.put("x2", String.valueOf(x2));
        return input;
    }
}
