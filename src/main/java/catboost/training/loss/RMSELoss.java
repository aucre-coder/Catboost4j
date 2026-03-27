package catboost.training.loss;

public class RMSELoss implements LossFunction {

    public void computeGradients(double[] predictions,
                                 double[] targets,
                                 double[] weights,
                                 double[] gradients,
                                 double[] hessians) {
        for (int i = 0; i < predictions.length; i++) {
            double weight = weights == null ? 1.0 : weights[i];
            gradients[i] = (targets[i] - predictions[i]) * weight;
            hessians[i] = weight;
        }
    }

    public double computeLoss(double[] predictions, double[] targets, double[] weights) {
        double weightedError = 0.0;
        double totalWeight = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double weight = weights == null ? 1.0 : weights[i];
            double residual = predictions[i] - targets[i];
            weightedError += weight * residual * residual;
            totalWeight += weight;
        }
        return Math.sqrt(weightedError / totalWeight);
    }
}
