import io
import logging
import os
import random
import time
from typing import List, Iterable

import keras.backend as K
import numpy as np
import pandas as pd
import twitter
import yaml
from injector import inject, Injector, singleton
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from keras.layers import Activation, Bidirectional, Dense, Embedding, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm


@singleton
class MyTwitterBot(object):
    @inject
    def __init__(self):
        config_path = os.getenv('MYBOT_TWITTER_CONFIG_PATH')
        if config_path is None:
            config_path = os.path.expanduser('~/.my-bot/config.yaml')

        print("Loading config from `{}`.".format(config_path))
        with io.open(config_path, 'r', encoding='utf8') as f:
            self._config = yaml.safe_load(f)

        self._logger = logging.getLogger('mybot.twitter')
        log_level = self._config.get('log level', logging.WARN)
        self._logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s:%(filename)s:%(funcName)s\n%(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        twitter_api_config = self._config['Twitter']
        self._twitter_api = twitter.Api(consumer_key=twitter_api_config['consumer key'],
                                        consumer_secret=twitter_api_config['consumer secret'],
                                        access_token_key=twitter_api_config['access token key'],
                                        access_token_secret=twitter_api_config['access token secret'])

        self._max_num_context_tokens = 10
        self._begin_token = '--bos--'
        self._eos_token = '--eos--'
        self._pad_index = 0

        self._tweets_path = os.path.expanduser(self._config['tweets path'])

        self._logger.info("Setting up tokenizer.")
        # TODO Keep punct as separate tokens.
        self._tokenizer = Tokenizer(filters='!"$%&()*+,./:;<=>?[\\]^`{|}~\t\n')
        tweets = self.get_tweets_for_training()
        self._tokenizer.fit_on_texts(tweets)

        self._embedding_dim = 300

    def get_tweets_for_training(self) -> Iterable[str]:
        result = map(self.pre_process, self.get_stored_tweets())
        result = map(lambda t: "{} {} {}".format(self._begin_token, t, self._eos_token), result)
        return result

    def get_stored_tweets(self):
        return pd.read_csv(self._tweets_path, encoding='utf-8').tweet

    def get_tweets(self, screen_name):
        result = set()
        max_id = None
        with tqdm(desc="Getting tweets",
                  unit_scale=True, mininterval=2, unit=" tweets") as pbar:
            while True:
                tweets: List[twitter.Status] = self._twitter_api.GetUserTimeline(screen_name=screen_name, max_id=max_id)
                if len(tweets) == 0:
                    break
                if max_id is None:
                    max_id = tweets[0].id
                for tweet in tweets:
                    max_id = min(max_id, tweet.id)
                    text = tweet.text.strip()
                    if len(text) > 2:
                        result.add(text)
                    pbar.update()
        self._logger.info("Found %d tweets.", len(result))

        pd.DataFrame(list(result), columns=['tweet']) \
            .to_csv(self._tweets_path, encoding='utf-8', index=False)

        return result

    def run(self):
        # FIXME Update for word based model.
        model = load_model('model.h5')

        while True:
            # Pick random tweet to start with.
            tweet = random.choice(old_tweets)

            sentence = tweet[:self.context_char_size]
            generated = sentence

            # 179 just in case.
            for i in range(179 - self.context_char_size):
                x_pred = np.zeros((1, self.context_char_size, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds)
                next_char = chars[next_index]

                if next_char == self._eos_char:
                    break

                generated += next_char
                sentence = sentence[1:] + next_char

            # self._twitter_api.PostUpdate(generated)
            self._logger.info("With seed: %s\n  Generated: \"%s\"",
                              generated[:self.context_char_size],
                              generated)
            # Wait 1 hour.
            # time.sleep(60 * 60)

    def pre_process(self, text):
        result = text
        if result.startswith("RT "):
            result = result[3:]
        result = result.lower()

        return result

    @staticmethod
    def sample(preds, temperature=1.0) -> int:
        """
        Sample an index from a probability array.
        """
        if temperature is None:
            preds = K.eval(K.softmax(preds))
            preds = np.random.multinomial(1, preds, 1)
        elif temperature != 'argmax':
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            preds = np.random.multinomial(1, preds, 1)

        return np.argmax(preds)

    def train(self):
        step = 1
        prefixes = []
        y = []

        tweets = self.get_tweets_for_training()
        sequences = self._tokenizer.texts_to_sequences_generator(tweets)
        token_index = self._tokenizer.word_index
        id2token = {index: token for (token, index) in token_index.items()}

        # Cache one-hot vectors for memory efficiency.
        token_one_hot_vectors = {}
        for tweet in tqdm(sequences,
                          desc="Building training data",
                          unit_scale=True, mininterval=2, unit=" tweets"):
            # Get sequences with just the first token,
            # and window through until we have sequences up to before the last token.
            for next_token_index in range(1, len(tweet), step):
                prefixes.append(tweet[max(0, next_token_index - b._max_num_context_tokens): next_token_index])
                token_vec = token_one_hot_vectors.get(tweet[next_token_index])
                if token_vec is None:
                    # Add 1 since token index starts at 1.
                    token_vec = np.zeros(len(token_index) + 1, dtype=np.int32)
                    token_vec[tweet[next_token_index]] = 1
                    token_one_hot_vectors[tweet[next_token_index]] = token_vec
                y.append(token_vec)

        prefixes = pad_sequences(prefixes, maxlen=self._max_num_context_tokens)
        y = np.array(y)

        self._logger.info("Setting up the model.")
        embeddings_index = {}
        # TODO Load embeddings right into matrix.
        with open(os.path.expanduser(self._config['embeddings path']), encoding='utf-8') as f:
            for line in tqdm(f,
                             desc="Loading embeddings",
                             unit_scale=True, mininterval=2, unit=" tokens"):
                values = line.split(maxsplit=1)
                token = values[0]
                if token in token_index:
                    coefs = np.asarray(values[1].split(), dtype=np.float32)
                    embeddings_index[token] = coefs

        # Add one for padding and since token index starts at 1.
        embeddings_matrix = np.zeros((len(token_index) + 1, self._embedding_dim))
        embeddings_matrix[self._pad_index] = np.zeros((self._embedding_dim,), dtype=np.float32)
        for i, token in enumerate(token_index, start=1):
            embedding_vec = embeddings_index.get(token)
            if embedding_vec is not None:
                embeddings_matrix[i] = embedding_vec
            else:
                # Make a new vector.
                # FIXME Make new vec with same magnitude as others.
                # FIXME Make new vec far from others.
                embeddings_matrix[i] = np.random.uniform(low=-0.5, high=0.5, size=(self._embedding_dim,)).astype(
                    np.float32)

        model = Sequential()
        model.add(Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],
                            weights=[embeddings_matrix],
                            input_length=self._max_num_context_tokens,
                            trainable=True))

        model.add(Bidirectional(LSTM(256, return_sequences=True,
                                     # dropout=0.1, recurrent_dropout=0.1
                                     )))
        model.add(Bidirectional(LSTM(256,
                                     # dropout=0.1, recurrent_dropout=0.1
                                     )))

        # model.add(Conv1D(filters=1,
        #                  kernel_size=16,
        #                  strides=1,
        #                  padding='causal',
        #                  input_shape=(x.shape[1], x.shape[2])))
        # model.add(Flatten())

        # model.add(Dropout(0.2))
        # Add 1 since token index starts at 1.
        model.add(Dense(y.shape[1]))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01, decay=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        model.summary()

        def on_epoch_end(epoch, logs):
            print()
            print("----- Generating text after epoch: {}".format(epoch))

            for diversity in ['argmax', None, 0.2, 0.5, 1.0, 1.2]:
                print("----- diversity: {}".format(diversity))

                sentence = [token_index[self._begin_token]]
                print("----- Generating with seed: \"{}\"".format(self._begin_token))
                for i in range(179):
                    x_pred = pad_sequences([sentence], maxlen=self._max_num_context_tokens)
                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = self.sample(preds, diversity)
                    next_token = id2token[next_index]

                    if next_token == self._eos_token:
                        break

                    sentence.append(next_index)
                    try:
                        print(next_token, end=" ")
                    except:
                        print("ERR", end=" ")
                print()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        # Mainly just do validation so that we can TensorBoard.
        validation_split = 0.02
        callbacks = [
            print_callback,
            ModelCheckpoint('model.h5', verbose=1)]
        if validation_split > 0:
            os.makedirs('tensorboard_logs', exist_ok=True)
            callbacks.append(TensorBoard('tensorboard_logs/{}'.format(int(time.time())),
                                         histogram_freq=1))
        model.fit(prefixes, y,
                  validation_split=validation_split,
                  verbose=1,
                  batch_size=128,
                  epochs=60,
                  callbacks=callbacks)


if __name__ == '__main__':
    injector = Injector()
    b: MyTwitterBot = injector.get(MyTwitterBot)
    # b.get_tweets('chelsdelaney11')
    b.train()
    # b.run()

    # with open('tweets.csv', encoding='utf-8') as f:
    #     text = f.read()
    # wordcloud = WordCloud(max_font_size=30).generate(text)
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()
    #
