package catboost.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Ordered feature names for float-only training.
 */
public class FeatureSchema {

    private final List<String> featureNames;

    public FeatureSchema(List<String> featureNames) {
        if (featureNames == null || featureNames.isEmpty()) {
            throw new IllegalArgumentException("featureNames must not be empty");
        }
        this.featureNames = Collections.unmodifiableList(new ArrayList<String>(featureNames));
    }

    public int size() {
        return featureNames.size();
    }

    public String getFeatureName(int index) {
        return featureNames.get(index);
    }

    public List<String> getFeatureNames() {
        return featureNames;
    }
}
