package catboost.training.tree;

public class SplitCandidate {

    private final int featureIndex;
    private final int borderIndex;
    private final double score;

    public SplitCandidate(int featureIndex, int borderIndex, double score) {
        this.featureIndex = featureIndex;
        this.borderIndex = borderIndex;
        this.score = score;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    public int getBorderIndex() {
        return borderIndex;
    }

    public double getScore() {
        return score;
    }
}
