# Personalization via Meta-Learning of Person Similarity Relationships

### Project Summary

Problem setup: In many machine learning problems, we consider data collected from multiple people (or domains). How can we best train models
  that are specialized for each individual person/domain, while leveraging commonalities across people/domains?

The repo contains code for running several methods of personalized machine learning:
* Proposed method: A meta-learning based approach for automatically learning how much to weight the data of each person/domain when training a model for a particular person/domain.
    We learn a person-wise similarity measure that is directly optimized to minimize training loss, and use this to determine how to weight each sample when training personalized models.
* Baselines: cluster similar people and train cluster-specific models, fine-tune person specific models, train person-specific models from scratch

### Running the Code 
1. Create config file <my_config>.json (see examples in src/argument_configs)
2. Run . ./path.sh to update your Python path
3. Run python runners/run_multi_exp.py <my_config.json>

