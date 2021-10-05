import nltk
import numpy as np
import os
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize

os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~') + "/.cache/tensorflow/"
import tensorflow_hub as hub
import tensorflow as tf

from transformers import AutoTokenizer, AutoModel

import torch
import logging

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.CRITICAL)

# stopwords = set(nltk.corpus.stopwords.words('english'))
_tokenizer_fn = English()


# _english.from_disk(os.path.join(os.path.dirname(spacy.__file__), "data/en_core_web_lg/en_core_web_lg-2.0.0"))
# from spacy.vocab import Vocab
# vocab = Vocab().from_disk(os.path.join(os.path.dirname(spacy.__file__), "data/en_core_web_lg/en_core_web_lg-2.0.0/vocab"))
# spacy.load("en")


def get_wordvectors(vocab: set):
    """
    Load the word vectors for a given vocabulary

    :param vocab: set of tokens
    :return: dict of token to vector
    """
    word_vectors = {}

    with open("/home/tebesu/ParlAI/data/models/glove_vectors/glove.6B.50d.txt") as f:
        for line in f:
            line = line.strip().split(' ')
            token = line[0].strip()
            if token in vocab:
                word_vectors[token] = np.asarray(line[1:], np.float32)
    return word_vectors


def tokenizer(text: str, method='split'):
    """
    Perform text tokenization with Spacy, split or nltk

    :param text:
    :param method: type of tokenization: spacy, whitespace split, nltk
    :return:
    """
    text = text.strip()
    if method == 'spacy':
        # return [token.lemma_ for token in _tokenizer_fn(text)]
        return [token.text for token in _tokenizer_fn(text)]
    elif method == 'split':
        return text.split(" ")
    elif method == 'nltk':
        return nltk.word_tokenize(text)
    else:
        raise ValueError("Unknown tokenizer method: supported are spacy, split or nltk")


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


class TextEncoder(object):

    def __init__(self, visible_devices: str,
                 bert_num: str,
                 encoder: str = 'transformer',
                 sess: tf.Session = None,
                 trainable: bool = False,
                 is_bert: bool = False):
        """
        Internally we use a universal sentence encoder
        """
        self._is_bert = encoder == 'bert' or is_bert

        # Initialize Text Encoder
        models = {
            'elmo': "https://tfhub.dev/google/elmo/2",
            # Transformer universal sentence encoder
            "transformer": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
            # Deep averaging network universal sentence encoder
            "dan": "https://tfhub.dev/google/universal-sentence-encoder/2",
            # Simple RNN Language Model
            'nnlm': "https://tfhub.dev/google/nnlm-en-dim128/1",

            # Bert Small
            'bert': "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",

            # Bert large
            # 'bert': "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1",
        }
        # Add it if we do not have it
        if encoder not in models:
            models[encoder] = encoder

        os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_devices)
        os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~') + "/.cache/tensorflow/"

        if self._is_bert:
            # ====================================================================
            # Start Pytorch Transformers
            # ====================================================================

            # Set model name here
            # Make sure the appropriate aggregation of output of each model is used
            # https://huggingface.co/sentence-transformers
            '''
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # R@10 ~ 2.5
            model_name = 'sentence-transformers/bert-base-nli-mean-tokens'  # R@10 ~ 1.5
            model_name = 'sentence-transformers/all-MiniLM-L12-v2'  # R@10 ~ 2.8
            model_name = 'sentence-transformers/all-mpnet-base-v2'  # R@10 ~ 2.6
            model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'  #
            model_name = 'sentence-transformers/paraphrase-MiniLM-L12-v2'  # R@10 ~ 2.6
            '''

            sent_bert = {
                '1': 'sentence-transformers/all-distilroberta-v1',
                '2': 'sentence-transformers/distilroberta-base-msmarco-v1',
                '3': 'sentence-transformers/distilroberta-base-msmarco-v2',
                '4': 'sentence-transformers/distilroberta-base-paraphrase-v1',
                '5': 'sentence-transformers/msmarco-distilroberta-base-v2',
                # model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'  #
                # model_name = 'sentence-transformers/paraphrase-MiniLM-L12-v2'  # R@10 ~ 2.6
                '6': "sentence-transformers/all-MiniLM-L6-v2",  # R@10 ~ 2.5,
                '7': 'sentence-transformers/bert-base-nli-mean-tokens',  # R@10 ~ 1.5,
                '8': 'sentence-transformers/all-MiniLM-L12-v2',  # R@10 ~ 2.8
                '9': 'sentence-transformers/all-mpnet-base-v2',  # R@10 ~ 2.6
                '10': 'sentence-transformers/paraphrase-mpnet-base-v2',  #
                '11': 'sentence-transformers/paraphrase-MiniLM-L12-v2',  # R@10 ~ 2.6

                '12': 'sentence-transformers/LaBSE',
                '13': 'sentence-transformers/bert-large-nli-mean-tokens',
                '14': 'sentence-transformers/bert-base-nli-max-tokens',
                '15': 'sentence-transformers/bert-base-nli-stsb-mean-tokens',
                '16': 'sentence-transformers/bert-large-nli-cls-token',
                '17': 'sentence-transformers/all-mpnet-base-v1',
                '18': 'sentence-transformers/all-mpnet-base-v2',
                '19': 'sentence-transformers/all-roberta-large-v1',
                '20': 'sentence-transformers/allenai-specter',
                '21': 'sentence-transformers/all-mpnet-base-v1',
                '22': 'sentence-transformers/all-mpnet-base-v1',
                '23': 'sentence-transformers/all-mpnet-base-v1',
                '24': 'sentence-transformers/all-mpnet-base-v1',
                '25': 'sentence-transformers/all-mpnet-base-v1',
                '26': 'sentence-transformers/all-mpnet-base-v1',
                '27': 'sentence-transformers/all-mpnet-base-v1',
                '28': 'sentence-transformers/all-mpnet-base-v1',
                '29': 'sentence-transformers/all-mpnet-base-v1',
                '30': 'sentence-transformers/all-mpnet-base-v1',

            }
            model_name = sent_bert[bert_num]

            print("Model name = ", model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            #self._model.cuda()
            self._model.eval()
            # ====================================================================
            # END Pytorch Transformers
            # ====================================================================
        else:
            if sess:
                self.sess = sess
            else:
                self.sess = tf.Session(
                    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self._embed_module = hub.Module(models[encoder], trainable=trainable)
            self._input_text = tf.placeholder(tf.string, shape=(None))
            self._encoded_text_op = self._embed_module(self._input_text,
                                                       as_dict=True)

            self.sess.run([tf.global_variables_initializer(),
                           tf.tables_initializer()])

    # def _encode_bert(self, text: str):
    #     """
    #     Use bert Pytorch version
    #
    #     TF version is messy
    #
    #     :param text:
    #     :return:
    #     """
    #     tokens = self.tokenizer.tokenize(text)[-510:]
    #
    #     # Add final EOS if required
    #     if not text.endswith(" [SEP]"):
    #         tokens.append("[SEP]")
    #
    #     tokens = ['[CLS]'] + tokens
    #
    #     # Convert token to vocabulary indices
    #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     segments_ids = [0] * len(token_ids)
    #     mask = [1] * len(token_ids)
    #     return self.sess.run(self._encoded_text_op, feed_dict={
    #         self._input_id: [token_ids],
    #         self._input_seg_id: [segments_ids],
    #         self._input_mask: [mask]
    #     })

    def encode(self, text, as_dict=False):
        """

        :param text: String of text or List of strings
        :param as_dict: Bool, return results as a dict or not
        :return: Encoded Vector from text
        """
        if self._is_bert:
            # ====================================================================
            # Start Pytorch Transformers
            # ====================================================================
            # TODO: Try nltk sent tokenizer?
            text = [sent.strip() for sent in text.split("[SEP]") if len(sent.strip())]
            # text = self.tokenizer([sent.strip() for sent in text.split("[SEP]") if len(sent.strip())]
            # sent_tokens = sent_tokenize(text)
            # print("sent_tokens using nltk = ", sent_tokens)
            inputs = self._tokenizer(text, return_tensors='pt', max_length=512,
                                     padding=True, truncation=True)
            with torch.no_grad():
                #inputs = {k: v.cuda() for k, v in inputs.items()}
                inputs = {k: v for k, v in inputs.items()}
                output = self._model(**inputs)
                # output = output.pooler_output.cpu()
                output = mean_pooling(output, inputs['attention_mask'])
                output = output.mean(dim=0, keepdim=True)
                output = output.cpu()
            if as_dict:
                return {'default': output}
            else:
                return output
            # ====================================================================
            # End Pytorch Transformers
            # ====================================================================
        if isinstance(text, str):
            text = [text]
        output = self.sess.run(self._encoded_text_op, {self._input_text: text})
        if as_dict:
            return output

        if 'embeddings' in output:
            return output['embeddings']
        return output['default']

