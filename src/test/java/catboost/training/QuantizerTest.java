package catboost.training;

import catboost.training.quantization.Quantizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizerTest {

    @Test
    void skipsBordersForConstantFeature() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {5.0},
                        {5.0},
                        {5.0}
                },
                new double[]{1.0, 1.0, 1.0},
                Arrays.asList("x")
        );

        QuantizedDataset quantized = new Quantizer().fit(dataset, new TrainerConfig().setMaxBins(4));

        assertEquals(0, quantized.getBorderCount(0));
        assertEquals(0, quantized.getBin(0, 0));
        assertEquals(0, quantized.getBin(0, 1));
        assertEquals(0, quantized.getBin(0, 2));
    }

    @Test
    void createsMonotonicBordersAndStableBins() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0},
                        {1.0},
                        {2.0},
                        {3.0},
                        {10.0},
                        {11.0},
                        {12.0}
                },
                new double[]{0, 0, 0, 0, 0, 0, 0},
                Arrays.asList("x")
        );

        QuantizedDataset quantized = new Quantizer().fit(dataset, new TrainerConfig().setMaxBins(3));

        double[] borders = quantized.getBorders(0);
        assertTrue(borders.length > 0);
        for (int i = 1; i < borders.length; i++) {
            assertTrue(borders[i] > borders[i - 1]);
        }
        assertArrayEquals(new double[]{2.5, 10.5}, borders, 1e-12);
        assertArrayEquals(new short[]{0, 0, 0, 1, 1, 2, 2}, quantized.getBinsForFeature(0));
    }

    @Test
    void matchesCatBoostWeightedUniqueGreedyLogSumCase() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {6.0},
                        {2.0},
                        {7.0},
                        {3.0},
                        {7.0},
                        {0.0},
                        {3.0},
                        {7.0},
                        {5.0}
                },
                new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8},
                Arrays.asList("x")
        );

        QuantizedDataset quantized = new Quantizer().fit(dataset, new TrainerConfig().setMaxBins(5));

        assertArrayEquals(new double[]{2.5, 4.0, 5.5, 6.5}, quantized.getBorders(0), 1e-12);
        assertArrayEquals(new short[]{3, 0, 4, 1, 4, 0, 1, 4, 2}, quantized.getBinsForFeature(0));
    }

    @Test
    void collapsesValuesThatBecomeEqualAfterCatBoostFloatCasting() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {16777216.2},
                        {16777216.4},
                        {16777218.0}
                },
                new double[]{0.0, 1.0, 2.0},
                Arrays.asList("x")
        );

        QuantizedDataset quantized = new Quantizer().fit(dataset, new TrainerConfig().setMaxBins(4));

        assertArrayEquals(new double[]{16777217.0}, quantized.getBorders(0), 1e-12);
        assertArrayEquals(new short[]{0, 0, 1}, quantized.getBinsForFeature(0));
    }
}
