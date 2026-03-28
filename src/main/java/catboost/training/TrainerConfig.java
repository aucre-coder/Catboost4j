package catboost.training;

/**
 * Training configuration for the native Java CatBoost parity subset:
 * CPU, RMSE, ordered boosting, symmetric trees, float features only.
 */
public class TrainerConfig {

    private int iterations = 100;
    private int depth = 6;
    private int maxBins = 255;
    private double learningRate = 0.03;
    private double l2LeafReg = 3.0;
    private double randomStrength = 1.0;
    private BootstrapType bootstrapType = BootstrapType.BAYESIAN;
    private double baggingTemperature = 1.0;
    private double subsample = 0.66;
    private long randomSeed = 0L;

    public int getIterations() {
        return iterations;
    }

    public TrainerConfig setIterations(int iterations) {
        if (iterations <= 0) {
            throw new IllegalArgumentException("iterations must be positive");
        }
        this.iterations = iterations;
        return this;
    }

    public int getDepth() {
        return depth;
    }

    public TrainerConfig setDepth(int depth) {
        if (depth <= 0 || depth > 16) {
            throw new IllegalArgumentException("depth must be in range [1, 16]");
        }
        this.depth = depth;
        return this;
    }

    public int getMaxBins() {
        return maxBins;
    }

    public TrainerConfig setMaxBins(int maxBins) {
        if (maxBins < 2 || maxBins > 255) {
            throw new IllegalArgumentException("maxBins must be in range [2, 255]");
        }
        this.maxBins = maxBins;
        return this;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public TrainerConfig setLearningRate(double learningRate) {
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("learningRate must be positive");
        }
        this.learningRate = learningRate;
        return this;
    }

    public double getL2LeafReg() {
        return l2LeafReg;
    }

    public TrainerConfig setL2LeafReg(double l2LeafReg) {
        if (l2LeafReg < 0.0) {
            throw new IllegalArgumentException("l2LeafReg must be non-negative");
        }
        this.l2LeafReg = l2LeafReg;
        return this;
    }

    public double getRandomStrength() {
        return randomStrength;
    }

    public TrainerConfig setRandomStrength(double randomStrength) {
        if (randomStrength < 0.0) {
            throw new IllegalArgumentException("randomStrength must be non-negative");
        }
        this.randomStrength = randomStrength;
        return this;
    }

    public BootstrapType getBootstrapType() {
        return bootstrapType;
    }

    public TrainerConfig setBootstrapType(BootstrapType bootstrapType) {
        if (bootstrapType == null) {
            throw new IllegalArgumentException("bootstrapType must not be null");
        }
        this.bootstrapType = bootstrapType;
        return this;
    }

    public double getBaggingTemperature() {
        return baggingTemperature;
    }

    public TrainerConfig setBaggingTemperature(double baggingTemperature) {
        if (baggingTemperature < 0.0) {
            throw new IllegalArgumentException("baggingTemperature must be non-negative");
        }
        this.baggingTemperature = baggingTemperature;
        return this;
    }

    public double getSubsample() {
        return subsample;
    }

    public TrainerConfig setSubsample(double subsample) {
        if (!(subsample > 0.0 && subsample <= 1.0)) {
            throw new IllegalArgumentException("subsample must be in range (0, 1]");
        }
        this.subsample = subsample;
        return this;
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    public TrainerConfig setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        return this;
    }

    void validateForParityScope() {
        if (bootstrapType == BootstrapType.BERNOULLI) {
            throw new IllegalArgumentException(
                    "bootstrap_type=Bernoulli is not supported in the strict native-Java CatBoost parity scope"
            );
        }
    }
}
