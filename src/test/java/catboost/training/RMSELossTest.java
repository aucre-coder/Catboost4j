package catboost.training;

import catboost.training.loss.RMSELoss;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RMSELossTest {

    @Test
    void computesGradientsAndHessiansWithWeights() {
        RMSELoss loss = new RMSELoss();
        double[] gradients = new double[3];
        double[] hessians = new double[3];

        loss.computeGradients(
                new double[]{3.0, 2.0, -1.0},
                new double[]{1.0, 5.0, 1.0},
                new double[]{1.0, 2.0, 0.5},
                gradients,
                hessians
        );

        assertArrayEquals(new double[]{2.0, -6.0, -1.0}, gradients, 1e-12);
        assertArrayEquals(new double[]{1.0, 2.0, 0.5}, hessians, 1e-12);
    }

    @Test
    void computesWeightedMseLoss() {
        RMSELoss loss = new RMSELoss();
        double value = loss.computeLoss(
                new double[]{3.0, 2.0},
                new double[]{1.0, 5.0},
                new double[]{1.0, 3.0}
        );

        assertEquals(Math.sqrt(7.75), value, 1e-12);
    }
}
