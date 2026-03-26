package catboost.training;

public class ObliviousSplit {

    private final int featureIndex;
    private final int borderIndex;

    public ObliviousSplit(int featureIndex, int borderIndex) {
        this.featureIndex = featureIndex;
        this.borderIndex = borderIndex;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    public int getBorderIndex() {
        return borderIndex;
    }
}
