package catboost.model;

import catboost.tree.TreeNode;

import java.util.List;
import java.util.Map;

/**
 * This class is used for predicting the output of Catboost Model
 * @author parasmalik
 */
public class Model {

    private final List<TreeNode> roots;
    private final Double scale;
    private final Double bias;
    public Model(List<TreeNode> roots, Double scale, Double bias){
        this.roots = roots;
        this.scale = scale;
        this.bias  = bias;
    }

    /**
     * This is for calculating the prediction from catboost model.
     * @param input in this map key is the name of the feature as given while training the model and value is String.valueOf(valueOfFeature)
     * @return prediction of given model on given input
     */
    public double predict(Map<String, String> input){
        return rawPredict(input, 0, roots.size());
    }

    /**
     * This method is used for to compute only subset of trees instead of all trees.
     * @param input in this map key is the name of the feature as given while training the model and value is String.valueOf(valueOfFeature)
     * @param startTree starting(including) index of the tree from where you want to compute the trees. Index start from 0
     * @param endTree ending(excluding) index of the tree
     * @return prediction of given model on given input
     */
    public double predict(Map<String, String> input, int startTree, int endTree){
        return rawPredict(input, startTree, endTree);
    }

    /**
     * Returns a probability by clamping the raw model output into the [0, 1] range.
     * Use this when the model was trained as a regression over binary targets such as no=0 and yes=1.
     */
    public double predictBoundedProbability(Map<String, String> input) {
        return clampProbability(predict(input));
    }

    /**
     * Returns a probability by clamping the raw subset prediction into the [0, 1] range.
     * Use this when the model was trained as a regression over binary targets such as no=0 and yes=1.
     */
    public double predictBoundedProbability(Map<String, String> input, int startTree, int endTree) {
        return clampProbability(predict(input, startTree, endTree));
    }

    /**
     * Returns a probability by applying the sigmoid transform to the raw model output.
     * Use this for binary classification models whose raw prediction is a logit.
     */
    public double predictSigmoidProbability(Map<String, String> input) {
        return sigmoid(predict(input));
    }

    /**
     * Returns a probability by applying the sigmoid transform to the raw subset prediction.
     * Use this for binary classification models whose raw prediction is a logit.
     */
    public double predictSigmoidProbability(Map<String, String> input, int startTree, int endTree) {
        return sigmoid(predict(input, startTree, endTree));
    }

    private double rawPredict(Map<String, String> input, int startTree, int endTree) {
        double result = 0.0;
        for(int i = startTree;i<endTree;i++){
            TreeNode root = roots.get(i);
            result += root.compute(input);
        }
        return result*scale + bias;
    }

    private double clampProbability(double value) {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }

    private double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

}
