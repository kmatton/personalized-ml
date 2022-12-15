"""
Factory functions.
"""
import os

from IPython import embed
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification, EarlyStoppingCallback, \
    DistilBertForSequenceClassification, DistilBertForMaskedLM

from clustering.cluster_data import KMeansModel
from dataset.amazon_reviews_clf_dataset import AmazonClfDataset
from dataset.amazon_reviews_mlm_dataset import AmazonMLMDataset
from dataset.mnist_dataset import MNISTDataset
from dataset.reddit_dataset import RedditDataset
from metrics.cross_entropy_loss import MaskedLMSampleWeightedCrossEntropy, MeanSampleCrossEntropy, CrossEntropy, \
    BinaryCrossEntropy, WeightedCrossEntropy, WeightedBinaryCrossEntropy
from metrics.metrics import ComputeMetrics
from models.basic_nn import BasicNN
from models.user_embed_nn import UserEmbeddingNN
from trainers.my_trainer import MyTrainer
from trainers.basic_trainer import BasicTrainer
from trainers.user_weight_trainer import UserWeightTrainer
from trainers.user_embed_trainer_meta import UserEmbedTrainerMeta
from trainers.user_weight_trainer_meta import UserWeightTrainerMeta


def get_dataset(data_args):
    dataset_init_args = [data_args.data_dir, data_args.raw_data_file]
    dataset_init_kwargs = dict(
        processed_data_dir=data_args.processed_data_dir,
        split_file=data_args.split_file,
        user_col=data_args.user_col,
        max_train_samples=data_args.max_train_samples,
        max_val_samples=data_args.max_val_samples,
        max_test_samples=data_args.max_test_samples,
        save_dataset=data_args.save_dataset
    )
    if data_args.dataset == "amazon_reviews_clf":
        dataset_init_args, dataset_init_kwargs = _add_language_dataset_args(dataset_init_args,
                                                                            dataset_init_kwargs,
                                                                            data_args)
        dataset = AmazonClfDataset(*dataset_init_args, **dataset_init_kwargs)
        return dataset
    elif data_args.dataset == "amazon_reviews_mlm":
        dataset_init_args, dataset_init_kwargs = _add_language_dataset_args(dataset_init_args,
                                                                            dataset_init_kwargs,
                                                                            data_args)
        dataset_init_kwargs = _add_mlm_dataset_args(dataset_init_kwargs, data_args)
        dataset = AmazonMLMDataset(*dataset_init_args, **dataset_init_kwargs)
        return dataset
    elif data_args.dataset == "reddit":
        dataset_init_args, dataset_init_kwargs = _add_language_dataset_args(dataset_init_args,
                                                                            dataset_init_kwargs,
                                                                            data_args)
        dataset_init_kwargs = _add_mlm_dataset_args(dataset_init_kwargs, data_args)
        dataset = RedditDataset(*dataset_init_args, **dataset_init_kwargs)
        return dataset
    elif data_args.dataset == "mnist":
        dataset = MNISTDataset(*dataset_init_args, **dataset_init_kwargs)
        return dataset
    print("ERROR: unrecognized dataset {}".format(data_args.dataset))
    print("Exiting...")
    exit(1)


def _add_language_dataset_args(dataset_init_args, dataset_init_kwargs, data_args):
    dataset_init_args += [data_args.tokenizer_name, data_args.tokenizer_cache_dir]
    dataset_init_kwargs["preprocessing_num_workers"] = data_args.preprocessing_num_workers
    return dataset_init_args, dataset_init_kwargs


def _add_mlm_dataset_args(dataset_init_kwargs, data_args):
    dataset_init_kwargs["fix_train_labels"] = data_args.fix_train_labels
    dataset_init_kwargs["fix_val_labels"] = data_args.fix_val_labels
    dataset_init_kwargs["fix_test_labels"] = data_args.fix_test_labels
    dataset_init_kwargs["mlm_probability"] = data_args.mlm_probability
    return dataset_init_kwargs


def get_embed_model(exp_args):
    if exp_args.embed_model_path is None:
        return None
    config = AutoConfig.from_pretrained(exp_args.embed_model_config)
    if exp_args.embed_model_type == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(exp_args.embed_model_path, config=config)
        return model
    elif exp_args.embed_model_type == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(exp_args.embed_model_path, config=config)
        return model
    print("ERROR: unrecognized embedding model {}".format(exp_args.embed_model_type))
    print("Exiting...")
    exit(1)


def get_model(model_args, dataset):
    config_name = model_args.config_name_or_path
    model_name = model_args.model_name_or_path
    if model_args.model == "sequence_classification":
        config = AutoConfig.from_pretrained(config_name, cache_dir=model_args.cache_dir, num_labels=dataset.d_out)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if model_args.n_freeze_layers:
            _freeze_layers(model_args.n_freeze_layers, model)
        return model
    elif model_args.model == "masked_lm":
        config = AutoConfig.from_pretrained(config_name, cache_dir=model_args.cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=model_args.cache_dir
        )
        return model
    elif model_args.model == "basic_nn":
        model = BasicNN(dataset.embed_dim, dataset.d_out, model_args.hidden_layer_sizes)
        if model_name is not None:  # load pre-trained weights
            print("loading pre-trained weights at path {}".format(model_name))
            loaded_weights = torch.load(model_name)
            if 'pre_classifier.weight' in loaded_weights.keys():  # loading DistilBERT model, just take final layers
                loaded_weights = _get_final_layers(loaded_weights)
            model.load_state_dict(loaded_weights)
        return model
    print("ERROR: unrecognized model {}".format(model_args.model))
    print("Exiting...")
    exit(1)


def get_user_embed_model(model_args, dataset):
    assert dataset.num_users is not None, "Need dataset to have users to use user embedding model"
    model_name = model_args.user_embed_model_name_or_path
    if model_args.user_embed_model == "basic_nn":
        model = BasicNN(dataset.num_users, model_args.user_embed_dim, model_args.user_embed_hidden_layer_sizes,
                        dropout_prob=0.0)
        if model_name is not None:
            print("loading pre-trained weights at path {}".format(model_name))
            model.load_state_dict(torch.load(model_name))
        return model
    elif model_args.user_embed_model == "user_embed_nn":
        model = UserEmbeddingNN(dataset.num_users, model_args.user_embed_dim, model_args.user_embed_hidden_layer_sizes,
                                dropout_prob=0.0)
        if model_name is not None:
            print("loading pre-trained weights at path {}".format(model_name))
            model.load_state_dict(torch.load(model_name))
        return model
    print("ERROR: unrecognized model {}".format(model_args.user_embed_model))
    print("Exiting...")
    exit(1)


def _get_final_layers(loaded_weights):
    keep_keys = ['pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias']
    remove_keys = [k for k in loaded_weights.keys() if k not in keep_keys]
    for key in remove_keys:
        del loaded_weights[key]
    loaded_weights['layers.0.weight'] = loaded_weights.pop('pre_classifier.weight')
    loaded_weights['layers.0.bias'] = loaded_weights.pop('pre_classifier.bias')
    loaded_weights['layers.3.weight'] = loaded_weights.pop('classifier.weight')
    loaded_weights['layers.3.bias'] = loaded_weights.pop('classifier.bias')
    return loaded_weights


def _freeze_layers(n_freeze_layers, model):
    if isinstance(model, DistilBertForSequenceClassification):
        # assume we also want to freeze embedding layer
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for i in range(n_freeze_layers):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
    else:
        print("ERROR: freezing layers is only implemented for DistilBertForSequence Classification")
        print("Exiting...")
        exit(1)


def get_loss_fn(loss_fn_name, model_config, model_args):
    loss_fct = None
    if loss_fn_name == "binary_cross_entropy":
        loss_fct = BinaryCrossEntropy()
    elif loss_fn_name == "weighted_binary_cross_entropy":
        loss_fct = WeightedBinaryCrossEntropy()
    elif loss_fn_name == "weighted_cross_entropy":
        loss_fct = WeightedCrossEntropy()
    elif loss_fn_name.endswith("cross_entropy"):
        if loss_fn_name.startswith("mean_sample"):
            loss_fct = MeanSampleCrossEntropy(model_config.vocab_size)
        elif loss_fn_name.startswith("weighted_sample"):
            loss_fct = MaskedLMSampleWeightedCrossEntropy(model_config.vocab_size)
        else:
            loss_fct = CrossEntropy()
    else:
        print("ERROR: unrecognized loss function {}".format(loss_fn_name))
        print("Exiting...")
        exit(1)
    if model_args.data_parallel:
        loss_fct = nn.DataParallel(loss_fct)
    return loss_fct


def get_cluster_model(exp_args):
    if exp_args.cluster_model == "kmeans":
        cluster_model = KMeansModel(exp_args.num_clusters, exp_args.cluster_feature)
        return cluster_model
    print("ERROR: unrecognized cluster model {}".format(exp_args.cluster_model))
    print("Exiting...")
    exit(1)


def get_trainer(exp_args, training_args, model_args, model, dataset, sample_weights=None):
    callbacks = None
    if exp_args.deterministic_mode:
        print("setting trainer to deterministic mode")
    loss_fn = get_loss_fn(exp_args.loss_fn, model.config, model_args)
    compute_metrics = None
    if len(exp_args.eval_metrics):
        compute_metrics = ComputeMetrics(exp_args.eval_metrics)
    if exp_args.trainer == "HFTrainer":
        if exp_args.early_stopping:
            callbacks = [
                EarlyStoppingCallback(early_stopping_patience=exp_args.es_patience,
                                      early_stopping_threshold=exp_args.es_threshold)]
            print("using early stopping")
        trainer_kwargs = dict(
            deterministic_mode=exp_args.deterministic_mode,
            use_weighted_loss=exp_args.use_weighted_loss,
            use_weighted_sampler=exp_args.use_weighted_sampler,
            train_sample_weights=sample_weights,
            loss_fn=loss_fn,
            model=model,
            args=training_args,
            train_dataset=dataset.train_data,
            eval_dataset=dataset.val_data,
            compute_metrics=compute_metrics,
            tokenizer=dataset.tokenizer,
            data_collator=dataset.get_data_collator(is_train=True),
            callbacks=callbacks
        )
        trainer = MyTrainer(**trainer_kwargs)
        return trainer
    if exp_args.trainer == "BasicTrainer":
        trainer_kwargs = _get_basic_trainer_args(model, loss_fn, dataset, training_args,
                                                 model_args, exp_args, compute_metrics)
        trainer = BasicTrainer(**trainer_kwargs)
        return trainer
    if exp_args.trainer == "UserWeightTrainer":
        basic_trainer_kwargs = _get_basic_trainer_args(model, loss_fn, dataset, training_args,
                                                       model_args, exp_args, compute_metrics)
        pred_trainer = BasicTrainer(**basic_trainer_kwargs)
        user_embed_model = get_user_embed_model(model_args, dataset)
        user_weight_trainer_kwargs = _get_user_weight_trainer_args(user_embed_model, dataset, pred_trainer, training_args, exp_args)
        trainer = UserWeightTrainer(**user_weight_trainer_kwargs)
        return trainer
    if exp_args.trainer == "UserWeightTrainerMeta":
        trainer = UserWeightTrainerMeta(users=dataset.users, user_col_name=dataset.user_col,
                                        seed=training_args.seed, output_dir=training_args.output_dir,
                                        prediction_model=model, train_dataset=dataset.train_data,
                                        val_dataset=dataset.val_data,
                                        train_collator=dataset.get_data_collator(is_train=True),
                                        val_collator=dataset.get_data_collator(is_train=False),
                                        pred_loss_fn=loss_fn, compute_metrics=compute_metrics,
                                        pred_lr=training_args.learning_rate, meta_lr=exp_args.meta_lr,
                                        user_batch_size=exp_args.user_batch_size, n_outer_it=exp_args.outer_steps)
        return trainer
    if exp_args.trainer == "UserEmbedTrainerMeta":
        user_embed_model = get_user_embed_model(model_args, dataset)
        trainer = UserEmbedTrainerMeta(users=dataset.users, user_col_name=dataset.user_col,
                                       seed=training_args.seed, output_dir=training_args.output_dir,
                                       prediction_model=model, user_embed_model=user_embed_model,
                                       init_user_embeds=dataset.user_embeddings, train_dataset=dataset.train_data,
                                       val_dataset=dataset.val_data,
                                       train_collator=dataset.get_data_collator(is_train=True),
                                       val_collator=dataset.get_data_collator(is_train=False),
                                       pred_loss_fn=loss_fn, compute_metrics=compute_metrics,
                                       pred_lr=training_args.learning_rate, meta_lr=exp_args.meta_lr,
                                       user_batch_size=exp_args.user_batch_size, n_outer_it=exp_args.outer_steps)
        return trainer
    print("ERROR: unrecognized trainer {}".format(exp_args.trainer))
    print("Exiting...")
    exit(1)


def _get_user_weight_trainer_args(user_embed_model, dataset, pred_trainer, training_args, exp_args):
    assert dataset.user_embeddings is not None, "cannot use user weight trainer without user embeddings"
    assert dataset.user_col is not None, "cannot use user weight trainer without user column"
    trainer_kwargs = dict(
        user_embed_model=user_embed_model,
        train_users=dataset.users,
        train_user_input_embeds=dataset.user_embeddings,
        pred_model_trainer=pred_trainer,
        seed=training_args.seed,
        user_col=dataset.user_col,
        train_dataset=dataset.train_data,
        val_dataset=dataset.val_data,
        lr_scheduler_type=training_args.lr_scheduler_type,
        weight_decay=training_args.weight_decay,
        adam_beta1=training_args.adam_beta1,
        adam_beta2=training_args.adam_beta2,
        adam_epsilon=training_args.adam_epsilon,
        learning_rate=training_args.learning_rate,
        warmup_steps=training_args.warmup_steps,
        warmup_ratio=training_args.warmup_ratio,
        user_embed_batch_size=exp_args.user_embed_batch_size,
        outer_steps=exp_args.outer_steps,
        inner_steps=exp_args.inner_steps,
        eval_steps=exp_args.user_embed_eval_steps,
        user_batch_size=exp_args.user_batch_size,
        output_dir=training_args.output_dir,
        early_stopping=exp_args.user_embed_early_stopping,
    )
    return trainer_kwargs


def _get_basic_trainer_args(model, loss_fn, dataset, training_args, model_args, exp_args, compute_metrics):
    trainer_kwargs = dict(
        model=model,
        loss_fn=loss_fn,
        seed=training_args.seed,
        train_dataset=dataset.train_data,
        val_dataset=dataset.val_data,
        train_collator=dataset.get_data_collator(is_train=True),
        val_collator=dataset.get_data_collator(is_train=False),
        data_parallel=model_args.data_parallel,
        optimizer_type=exp_args.optimizer_type,
        lr_scheduler_type=training_args.lr_scheduler_type,
        weight_decay=training_args.weight_decay,
        adam_beta1=training_args.adam_beta1,
        adam_beta2=training_args.adam_beta2,
        adam_epsilon=training_args.adam_epsilon,
        learning_rate=training_args.learning_rate,
        warmup_steps=training_args.warmup_steps,
        warmup_ratio=training_args.warmup_ratio,
        batch_size_per_device=training_args.per_device_train_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        compute_metrics=compute_metrics,
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        output_dir=training_args.output_dir,
        early_stopping=exp_args.early_stopping,
        es_patience=exp_args.es_patience,
        es_threshold=exp_args.es_threshold,
        load_best_model_at_end=training_args.load_best_model_at_end,
        deterministic_mode=exp_args.deterministic_mode,
        drop_last=training_args.dataloader_drop_last,
        max_grad_norm=training_args.max_grad_norm,
        verbose=exp_args.verbose
    )
    return trainer_kwargs
