# CatBoost4j
[IMPORTANT : APIs is not final, current code is yet not released]

This is pure Java prediction client of Catboost Machine learning library. It works on json format of the model. You can save the model to json format using 

*model.save_model("file_name", format = "json", pool = pool)*

# This library is tested till catboost version '0.21'

# Code Documentation is available at http://parasmalik.blogspot.com/2020/07/explanation-of-json-model-format-of.html

# Current Limitations : 
 Currently this repository can predict regression models from CatBoost JSON and can train experimental float-only RMSE regression models in pure Java.

# Experimental Training API
The repository now includes a first-cut trainer for:

- float features only
- RMSE / regression only
- in-memory model training to the existing `Model` prediction API

Example:

```java
Dataset dataset = Dataset.of(features, targets, Arrays.asList("x1", "x2"));
TrainerConfig config = new TrainerConfig()
        .setIterations(100)
        .setDepth(6)
        .setLearningRate(0.03)
        .setMaxBins(32)
        .setL2LeafReg(3.0);

Model model = new CatBoostTrainer(config).fit(dataset);
```

Non-goals of this first training cut:

- categorical feature training
- classification / multiclass
- CatBoost JSON export for newly trained models

# Versioning 
 We will keep the major and minor versions same as catboost package. So, if this lib able to predict till catboost version 0.21 we will keep version as 0.21.x.x 
 
# Contribution 
 Please feel free to suggest/implement/send Pull request. 
 
# Contact
In case if you are interested in the project or if you have questions, please contact with me by email: paras_malik_mca_iit@yahoo.com
