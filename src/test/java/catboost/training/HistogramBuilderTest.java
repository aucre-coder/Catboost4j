package catboost.training;

import catboost.training.quantization.Quantizer;
import catboost.training.tree.Histogram;
import catboost.training.tree.HistogramBuilder;
import catboost.training.tree.SplitCandidate;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class HistogramBuilderTest {

    @Test
    void findsBestSplitOnInformativeFeature() {
        Dataset dataset = Dataset.of(
                new double[][]{
                        {0.0, 10.0},
                        {0.1, 10.0},
                        {1.0, 10.0},
                        {1.1, 10.0}
                },
                new double[]{0.0, 0.0, 1.0, 1.0},
                Arrays.asList("x1", "x2")
        );
        QuantizedDataset quantized = new Quantizer().fit(dataset, new TrainerConfig().setMaxBins(4));
        HistogramBuilder histogramBuilder = new HistogramBuilder();
        int[] leafIndexes = new int[]{0, 0, 0, 0};
        double[] gradients = new double[]{0.0, 0.0, -1.0, -1.0};
        double[] hessians = new double[]{1.0, 1.0, 1.0, 1.0};

        Histogram informativeHistogram = histogramBuilder.build(quantized, 0, 1, leafIndexes, gradients, hessians);
        Histogram flatHistogram = histogramBuilder.build(quantized, 1, 1, leafIndexes, gradients, hessians);
        SplitCandidate informative = histogramBuilder.findBestSplit(0, informativeHistogram, 3.0);
        SplitCandidate flat = histogramBuilder.findBestSplit(1, flatHistogram, 3.0);

        assertNotNull(informative);
        assertNull(flat);
        assertEquals(0, informative.getFeatureIndex());
        assertEquals(1, informative.getBorderIndex());
        assertEquals(0.8, informative.getScore(), 1e-12);
    }
}
