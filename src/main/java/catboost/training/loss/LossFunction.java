package catboost.training.loss;

public interface LossFunction {

    void computeGradients(double[] predictions,
                          double[] targets,
                          double[] weights,
                          double[] gradients,
                          double[] hessians);

    double computeLoss(double[] predictions, double[] targets, double[] weights);
}
