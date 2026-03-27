package catboost.training;

/**
 * Quantized float features stored feature-major.
 */
public class QuantizedDataset {

    private final short[][] bins;
    private final double[][] borders;
    private final FeatureSchema featureSchema;
    private final int rowCount;

    public QuantizedDataset(short[][] bins, double[][] borders, FeatureSchema featureSchema, int rowCount) {
        this.bins = bins;
        this.borders = borders;
        this.featureSchema = featureSchema;
        this.rowCount = rowCount;
    }

    public int getFeatureCount() {
        return bins.length;
    }

    public int getRowCount() {
        return rowCount;
    }

    public short getBin(int featureIndex, int rowIndex) {
        return bins[featureIndex][rowIndex];
    }

    public short[] getBinsForFeature(int featureIndex) {
        short[] source = bins[featureIndex];
        short[] copy = new short[source.length];
        System.arraycopy(source, 0, copy, 0, source.length);
        return copy;
    }

    public double[] getBorders(int featureIndex) {
        double[] source = borders[featureIndex];
        double[] copy = new double[source.length];
        System.arraycopy(source, 0, copy, 0, source.length);
        return copy;
    }

    public int getBorderCount(int featureIndex) {
        return borders[featureIndex].length;
    }

    public FeatureSchema getFeatureSchema() {
        return featureSchema;
    }
}
