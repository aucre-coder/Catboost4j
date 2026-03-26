package catboost.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ObliviousTree {

    private final List<ObliviousSplit> splits;
    private final double[] leafValues;

    public ObliviousTree(List<ObliviousSplit> splits, double[] leafValues) {
        this.splits = Collections.unmodifiableList(new ArrayList<ObliviousSplit>(splits));
        this.leafValues = copy(leafValues);
    }

    public List<ObliviousSplit> getSplits() {
        return splits;
    }

    public double[] getLeafValues() {
        return copy(leafValues);
    }

    public int getDepth() {
        return splits.size();
    }

    public double getLeafValue(int leafIndex) {
        return leafValues[leafIndex];
    }

    private static double[] copy(double[] values) {
        double[] copy = new double[values.length];
        System.arraycopy(values, 0, copy, 0, values.length);
        return copy;
    }
}
