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
                        {1.0},
                        {2.0},
                        {2.0},
                        {3.0},
                        {4.0},
                        {10.0}
                },
                new double[]{0, 0, 0, 0, 0, 0},
                Arrays.asList("x")
        );

        QuantizedDataset quantized = new Quantizer().fit(dataset, new TrainerConfig().setMaxBins(4));

        double[] borders = quantized.getBorders(0);
        assertTrue(borders.length > 0);
        for (int i = 1; i < borders.length; i++) {
            assertTrue(borders[i] > borders[i - 1]);
        }
        assertArrayEquals(new double[]{1.5, 2.5, 3.5}, borders, 1e-12);
        assertArrayEquals(new short[]{0, 1, 1, 2, 3, 3}, quantized.getBinsForFeature(0));
    }
}
