package catboost.training.adapter;

import catboost.condition.FloatCondition;
import catboost.model.Model;
import catboost.training.ObliviousSplit;
import catboost.training.ObliviousTree;
import catboost.training.TrainingResult;
import catboost.tree.TreeNode;

import java.util.ArrayList;
import java.util.List;

public class ModelAdapter {

    public Model toModel(TrainingResult trainingResult) {
        List<TreeNode> roots = new ArrayList<TreeNode>();
        for (ObliviousTree tree : trainingResult.getTrees()) {
            roots.add(buildTreeNode(tree, trainingResult.getFeatureSchema(), trainingResult.getBorders(), 0, 0));
        }
        return new Model(roots, 1.0, trainingResult.getBias());
    }

    private TreeNode buildTreeNode(ObliviousTree tree,
                                   catboost.training.FeatureSchema featureSchema,
                                   double[][] borders,
                                   int splitDepth,
                                   int leafIndex) {
        if (splitDepth == tree.getDepth()) {
            return new TreeNode(tree.getLeafValue(leafIndex), leafIndex);
        }

        List<ObliviousSplit> splits = tree.getSplits();
        ObliviousSplit split = splits.get(splits.size() - 1 - splitDepth);
        String featureName = featureSchema.getFeatureName(split.getFeatureIndex());
        double border = borders[split.getFeatureIndex()][split.getBorderIndex()];
        return new TreeNode(
                new FloatCondition(featureName, border),
                buildTreeNode(tree, featureSchema, borders, splitDepth + 1, leafIndex << 1),
                buildTreeNode(tree, featureSchema, borders, splitDepth + 1, (leafIndex << 1) + 1)
        );
    }
}
