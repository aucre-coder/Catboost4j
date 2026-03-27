package catboost.training;

/**
 * Training configuration for the first-cut float-only regression trainer.
 */
public class TrainerConfig {

    private int iterations = 100;
    private int depth = 6;
    private int maxBins = 32;
    private double learningRate = 0.03;
    private double l2LeafReg = 3.0;
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

    public long getRandomSeed() {
        return randomSeed;
    }

    public TrainerConfig setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        return this;
    }
}
