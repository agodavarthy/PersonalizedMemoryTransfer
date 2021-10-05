import numpy as np
from spacy.lang.en import English
import nltk
import logging
import os
# from util.preprocess import TextEncoder
# os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~') + "/.cache/tensorflow/"
import tensorflow_hub as hub
import tensorflow as tf
# from util.helper import parser_add_str2bool
import argparse
import pandas as pd
import logging
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)
import shutil, glob, json
from tensorflow.python.estimator.canned.head import _BinaryLogisticHeadWithSigmoidCrossEntropyLoss
from functools import partial
# from rutil.util.tfutil import print_parameter_analysis, set_logging_config
from tensorflow.python.estimator.estimator import _load_global_step_from_checkpoint_dir


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser_add_str2bool(parser)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str,
                    default="0")

parser.add_argument('-enc', '--encoder', help='Type of text encoder to use',
                    type=str, required=True,
                    choices=['transformer', 'elmo', 'dan', 'nnlm', 'bert'])
parser.add_argument('-m', '--model_dir', help='model directory to save',
                    type=str, default=None)
parser.add_argument('-i', '--iters', help='Max iters', type=int, default=20)
parser.add_argument('-bs', '--batch_size', help='Batch Size', type=int, default=16)
parser.add_argument('-d', '--dataset', help='directory to dataset',
                    default='data/sentiment/sst/', type=str)
parser.add_argument('-l2', '--l2', help='L2 regularization', type=float, default=0.0)
parser.add_argument('-flr', '--fine_tune_lr', help='Learning rate for finetuning'
                                                   ' the embeddings',
                    type=float,
                    default=1e-6)
parser.add_argument('-max', '--max_length',
                    help='Truncate examples to max length, 0 = disable',
                    type=int, default=100)


def load_sst_data(filename, truncate=None):
    """
    Load Stanford Sentiment data

    :param filename: path to file
    :param truncate: Max length to truncate at
    :return: DataFrame with columns, target and text
    """
    data = []
    with open(filename) as f:
        # Skip header
        f.readline()
        for line in f:
            label, sent = line.strip().split(",", 1)
            if truncate:
                sent = " ".join(sent.split()[:truncate])
            data.append({'label': float(label), 'text': sent})
    df = pd.DataFrame.from_dict(data)
    df['label'] = df['label'].astype(np.float32)
    return df


def _add_metrics(labels, predictions, features, config):
    """
    Add additional metrics we want to track

    :param labels:
    :param predictions:
    :param features:
    :param config:
    :return:
    """
    # tf.round(predictions['predictions'])
    cls_id = predictions['class_ids']
    return {'accuracy': tf.metrics.accuracy(labels, cls_id),
            'auc': tf.metrics.auc(labels, cls_id),
            'precision': tf.metrics.precision(labels, cls_id),
            'recall': tf.metrics.recall(labels, cls_id)}


def bert_tokenizer():
    import sys
    sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
    from util.bert_tokenizer import FullTokenizer

    with tf.Graph().as_default():
        bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

def _model_fn(features, labels, mode, params=None, config=None):
    """

    :param features:
    :param labels:
    :param mode:
    :param params:
    :param config:
    :return:
    """

    tf.logging.info("\nFeatures: {}\nLabels: {}\n".format(features, labels))
    model_urls = {
        'elmo': "https://tfhub.dev/google/elmo/2",
        # Transformer universal sentence encoder
        "transformer": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
        # Deep averaging network universal sentence encoder
        "dan": "https://tfhub.dev/google/universal-sentence-encoder/2",
        # Simple RNN Language Model
        'nnlm': "https://tfhub.dev/google/nnlm-en-dim128/1",

        # Bert Small
        # 'bert': "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",

        # Bert large
        'bert': "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1",
    }
    # Remove labels
    if mode == tf.estimator.ModeKeys.PREDICT:
        labels = None
    # if (mode == tf.estimator.ModeKeys.TRAIN or
    #         mode == tf.estimator.ModeKeys.EVAL):

    head = _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(
            loss_reduction=tf.losses.Reduction.MEAN)
    # tf.estimator.BinaryClassHead()
    # head = tf.contrib.estimator.logistic_regression_head()
    embed = hub.Module(model_urls[params['encoder']],
                       trainable=params['fine_tune_lr'] > 0.0)
    if params['encoder'] == 'bert':
        inputs = embed(dict(
                input_ids=tf.cast(features['token_id'], tf.int32),
                input_mask=tf.cast(features['mask'], tf.int32),
                segment_ids=tf.cast(features['segment_id'], tf.int32)),
                signature="tokens", as_dict=True)["pooled_output"]
    else:
        inputs = embed(features['text'])

    output_layer = tf.layers.Dense(1, activation=None, use_bias=True,
                                   kernel_initializer=tf.glorot_uniform_initializer(),
                                   name='Output', kernel_constraint=None)
    logits = output_layer(inputs)
    train_op_fn = None

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer()
        if params['fine_tune_lr'] > 0.0:
            finetune_optimizer = tf.train.AdamOptimizer(learning_rate=params['fine_tune_lr'])

        def wrap_optimizer(loss_op):
            """Wrap function so we can use two optimizers with different learning rates"""
            optimize_op = optimizer.minimize(loss_op,
                                             global_step=tf.train.get_or_create_global_step(),
                                             var_list=output_layer.trainable_variables)
            if params['fine_tune_lr'] > 0.0:
                finetune_op = finetune_optimizer.minimize(loss_op,
                                                          global_step=tf.train.get_or_create_global_step(),
                                                          var_list=list(embed.variable_map.values()))
                return tf.group(optimize_op, finetune_op)
            else:
                return optimize_op

        train_op_fn = wrap_optimizer
    reg_loss = None

    if params['l2'] > 0:
        reg_loss = [params['l2'] * tf.nn.l2_loss(w) for w in tf.trainable_variables()]
        # reg_loss = tf.contrib.layers.apply_regularization([
        #     tf.contrib.layers.l2_regularizer(params['l2'])
        # ], )

    estimator_spec = head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            regularization_losses=reg_loss,
            #optimizer=optimizer,
            train_op_fn=train_op_fn,
            logits=logits)
    estimator_spec.predictions['embeddings'] = inputs

    # print_parameter_analysis(log_once=True)
    return estimator_spec


def _cast_dict_dtypes(results_dict):
    def cast_dtypes(value):
        """
        Cast to native python types for json serialization

        :param value:
        :return:
        """
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int64, np.int32)):
            return int(value)
        else:
            return value

    return {k: cast_dtypes(v) for k, v in results_dict.items()}


def serving_input_receiver_fn(bert=False):
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    if bert:

        inputs = {
            'token_id': tf.placeholder(tf.int32, [1, None]),
            'segment_id': tf.placeholder(tf.int32, [1, None]),
            'mask': tf.placeholder(tf.int32, [1, None])
        }
        receiver_tensors = inputs
    else:

        inputs = tf.placeholder(tf.string, shape=(None))
        # df['token_id'] = token_ids
        # df['segment_id'] = segment_ids
        # df['mask'] = masks
        receiver_tensors = {'text': inputs}

    return tf.estimator.export.ServingInputReceiver(inputs, receiver_tensors)
    # estimator.export_saved_model('saved_model', serving_input_receiver_fn)

def bert_preprocess(df, tokenizer, max_len):
    token_ids = np.zeros((len(df), max_len), dtype=np.float32)
    segment_ids = np.zeros((len(df), max_len), dtype=np.float32)
    masks = np.zeros((len(df), max_len), dtype=np.float32)

    for i, text in enumerate(df['text'].values):

        tokens = ['[CLS]'] + tokenizer.tokenize(text)[:max_len-2] + ['[SEP]']
        # tokens = [tokenizer.tokenize(s)[-510:] for s in text]
        # Convert token to vocabulary indices
        ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids[i, :len(ids)] = ids
        masks[i, :len(ids)] = 1

    return {'segment_id': segment_ids,
            'mask': masks,
            'token_id': token_ids
            }
    # np.array(mask, dtype=np.int64)
    # token_ids.append(np.array(ids, dtype=np.int64))
    # segment_ids.append(np.array(seg, dtype=np.int64))
    # masks.append(np.array(mask, dtype=np.int64))
    # return df


if __name__ == '__main__':
    opt = parser.parse_args()

    print('1')
    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                 #allow_soft_placement=True
                                 )
    exporters = None

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~') + "/.cache/tensorflow/"
    data_dir = opt.dataset
    # Data Loading...

    df_test = load_sst_data(os.path.join(data_dir, 'test_binary_sent.csv'), opt.max_length)
    df_train = load_sst_data(os.path.join(data_dir, 'train_binary_sent.csv'), opt.max_length)
    # df_valid = load_sst_data(os.path.join(data_dir, 'dev_binary_sent.csv'), opt.max_length)
    if opt.encoder == 'bert':
        print("Loading tokenizer")
        tokenizer = bert_tokenizer()
        print("Preprocessing testing data")
        test_bert = bert_preprocess(df_test, tokenizer, opt.max_length)
        print("Preprocessing training data")
        train_bert = bert_preprocess(df_train, tokenizer, opt.max_length)
        print("DONE!")

    num_steps = len(df_train) // opt.batch_size
    run_config = tf.estimator.RunConfig(model_dir=opt.model_dir,
                                        save_checkpoints_steps=num_steps-1,
                                        session_config=sess_config,
                                        # save_checkpoints_secs=_USE_DEFAULT,
                                        # log_step_count_steps=0,
                                        # save_summary_steps=100,
                                        keep_checkpoint_max=1)
    estimator = tf.estimator.Estimator(_model_fn,
                                       params=opt.__dict__,
                                       config=run_config)

    best_dir = os.path.join(estimator.model_dir, 'best')
    # estimator = tf.estimator.add_metrics(estimator, _add_metrics)

    # Save hyperparameters
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)

    # set_logging_config(opt.model_dir)
    with open(f"{opt.model_dir}/opt.json", 'w') as f:
        json.dump(opt.__dict__, f, sort_keys=True, indent=2)
    tf.logging.info(json.dumps(opt.__dict__, sort_keys=True, indent=2))

    # Build input producers
    if opt.encoder == 'bert':
        input_fn_train = tf.estimator.inputs.numpy_input_fn(
                train_bert, df_train['label'], batch_size=opt.batch_size,
                num_epochs=None,
                shuffle=True)
    else:
        input_fn_train = tf.estimator.inputs.pandas_input_fn(
                df_train, df_train['label'], batch_size=opt.batch_size,
                num_epochs=None,
                shuffle=True)
    # input_fn_pred_valid = tf.estimator.inputs.pandas_input_fn(
    #         df_valid, df_valid['label'], batch_size=opt.batch_size,
    #         shuffle=False)



    # Train/Evaluation Loops
    # train_spec = tf.estimator.TrainSpec(input_fn=input_fn_train,
    #                                     max_steps=num_steps)
    # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_pred_valid,
    #                                   exporters=exporters,
    #                                   #       start_delay_secs: Int. Start evaluating after waiting for this many
    #                                   #         seconds.
    #                                   #       throttle_secs: Int. Do not re-evaluate unless the last evaluation was
    #                                   #         started at least this many seconds ago. Of course, evaluation does not
    #                                   #         occur if no new checkpoints are available, hence, this is the minimum.
    #                                   start_delay_secs=60,
    #                                   throttle_secs=30, #name='valid'
    #                                   )

    tf.logging.info("\n\nStarting train and evaluate....\n")

    global_steps = _load_global_step_from_checkpoint_dir(estimator.model_dir)

    cur_iters = int(float(global_steps) // num_steps)
    iters = max(int(opt.iters - cur_iters), 0)

    tf.logging.info("Completed: {}".format(cur_iters))
    tf.logging.info("Iterations to run: {} / {}".format(iters, opt.iters))
    if opt.encoder == 'bert':
        input_fn_pred_train = tf.estimator.inputs.numpy_input_fn(
                train_bert, df_train['label'], batch_size=opt.batch_size,
                shuffle=False, queue_capacity=100)
        input_fn_pred_test = tf.estimator.inputs.numpy_input_fn(
                test_bert, df_test['label'], batch_size=opt.batch_size,
                shuffle=False, queue_capacity=100)
    else:
        input_fn_pred_train = tf.estimator.inputs.pandas_input_fn(
                df_train, df_train['label'], batch_size=opt.batch_size,
                shuffle=False, queue_capacity=100)
        input_fn_pred_test = tf.estimator.inputs.pandas_input_fn(
                df_test, df_test['label'], batch_size=opt.batch_size,
                shuffle=False, queue_capacity=100)

    best_accuracy = 0.0
    best_results = {}
    n_steps = num_steps * iters

    print()
    print(f"Total Steps per a iteration: {num_steps:,}")
    print(f"Running for {n_steps:,}")
    print()
    for i in range(2):
        if n_steps < 2:
            break
        estimator.train(input_fn_train, steps=n_steps//2)
        # Results
        results_train = estimator.evaluate(input_fn_pred_train, name='train')
        results_test = estimator.evaluate(input_fn_pred_test, name='test')
        #results_valid = estimator.evaluate(input_fn_pred_valid, name='valid')
        # if results_valid['accuracy'] > best_accuracy:
        #     tf.logging.info("New Best Accuracy: {:<10.4f} previous {:.4f}".format(
        #             results_valid['accuracy'], best_accuracy))
        #
        #     # Delete old
        #     if os.path.exists(best_dir):
        #         shutil.rmtree(best_dir)
        #     os.mkdir(best_dir)
        #     os.system("cp {}* {}".format(estimator.latest_checkpoint(), best_dir))
        #     best_accuracy = results_valid['accuracy']
        #     best_results = {'train': _cast_dict_dtypes(results_train),
        #                     'test': _cast_dict_dtypes(results_test),
        #                     'valid': _cast_dict_dtypes(results_valid)}
        #     json.dump(best_results,
        #               open(os.path.join(best_dir, 'results.json'), 'w'))

        tf.logging.info("\n\n\n")
        for k in results_train.keys():
            if k in {'global_step'}:
                continue
            tf.logging.info("{:<25} Train / Test:  {:.4f} / {:.4f}".format(
                    k, results_train[k], results_test[k]))
        tf.logging.info("")
    tf.logging.info("\n"*5)
    tf.logging.info("="*80)
    tf.logging.info("Creating Hub Module")
    tf.logging.info("=" * 80)
    # tf.logging.info("\n\n\nBest Results:")
    # best_results = json.load(open(os.path.join(best_dir, 'results.json')))
    # results_train, results_valid, results_test = best_results['train'], best_results['valid'], best_results['test']
    # for k in results_train.keys():
    #     if k in {'global_step'}:
    #         continue
    #     tf.logging.info("{:<25} Train / Test:  {:.4f} / {:.4f}".format(
    #             k, results_train[k], results_test[k]))

    def _module_fn():

        if opt.encoder == 'bert':
            inputs = {
                'token_id': tf.placeholder(tf.int32, [1, None]),
                'segment_id': tf.placeholder(tf.int32, [1, None]),
                'mask': tf.placeholder(tf.int32, [1, None])
            }
        else:
            inputs = {
                'text': tf.placeholder(dtype=tf.string, shape=[None])
            }

        estimator_spec = _model_fn(inputs, labels=None,
                                   mode=tf.estimator.ModeKeys.PREDICT,
                                   params=opt.__dict__,
                                   config=run_config)

        hub.add_signature(inputs=inputs, outputs={
            'class_ids': estimator_spec.predictions['class_ids'],
            'logistic': estimator_spec.predictions['logistic'],
            'embeddings': estimator_spec.predictions['embeddings']
        })

    # Get the checkpoint version we copied
    checkpoint = glob.glob("{}/model.ckpt-*.meta".format(run_config.model_dir   ))[0][:-5]
    # checkpoint = glob.glob("{}/model.ckpt-*.meta".format(best_dir))[0][:-5]

    tf.logging.info(f"\n\n" + ("="*80) + "Using Checkpoint: {checkpoint}\n" + ("="*80))
    module_spec = hub.create_module_spec(_module_fn)
    module_spec.export(os.path.join(run_config.model_dir, 'module'),
                       checkpoint_path=checkpoint)
    tf.logging.info("\n\n\nDONE!\n")

