import io
import logging
import os
import random
import sys
from typing import List, Iterable

import numpy as np
import pandas as pd
import twitter
import yaml
from injector import inject, Injector, singleton
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import Activation, Bidirectional, Dense, Embedding, GRU
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
        self._unk = '--unk--'
        self._pad_index = 0

        self._tweets_path = os.path.expanduser(self._config['embeddings path'])

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
        return pd.read_csv(self._tweets_path, encoding='utf-8', error_bad_lines=False).tweets

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
        old_tweets = self.get_stored_tweets()

        with open('chars.txt', encoding='utf-8') as f:
            chars = f.read()
        char_indices = dict((c, i) for i, c in enumerate(chars))

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
        if temperature is not None:
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
        else:
            probas = preds
        return np.argmax(probas)

    def train(self):

        step = 1
        prefixes = []
        next_tokens = []

        tweets = self.get_tweets_for_training()
        sequences = self._tokenizer.texts_to_sequences_generator(tweets)

        for tweet in sequences:
            # Get sequences with just the first token,
            # and window through until we have sequences up to before the last token.
            for start_index in range(-self._max_num_context_tokens + 1, len(tweet) - self._max_num_context_tokens,
                                     step):
                prefixes.append(tweet[max(0, start_index): start_index + self._max_num_context_tokens])
                next_tokens.append(tweet[start_index + self._max_num_context_tokens + 1])

        prefixes = pad_sequences(prefixes)

        self._logger.info("Setting up the model.")
        self._logger.info("Loading embeddings.")
        embeddings_index = {}
        with open(os.path.expanduser(self._config['embeddings path'])) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        token_index = self._tokenizer.word_index
        num_tokens = len(token_index)
        self._logger.info("Found %s unique tokens.", num_tokens)

        # Add one for padding.
        embeddings_matrix = np.zeros((num_tokens + 1, self._embedding_dim))
        embeddings_matrix[self._pad_index] = np.zeros((self._embedding_dim,), dtype=np.float32)
        for i, token in enumerate(token_index, start=1):
            embedding_vec = embeddings_index.get(token)
            if embedding_vec is not None:
                embeddings_matrix[i] = embedding_vec
            else:
                # Make a new vector.
                embeddings_matrix[i] = np.random.uniform(low=-0.5, high=0.5, size=(self._embedding_dim,)).astype(
                    np.float32)

        embedding_layer = Embedding(num_tokens, self._embedding_dim,
                                    weights=[embeddings_matrix],
                                    input_length=self._max_num_context_tokens,
                                    trainable=True)

        model = Sequential()
        model.add(embedding_layer)

        model.add(Bidirectional(GRU(32, return_sequences=True,
                                    # dropout=0.1, recurrent_dropout=0.1
                                    ),
                                input_shape=(self.context_char_size, len(chars))
                                ))
        # model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(32,
                                    # dropout=0.1, recurrent_dropout=0.1
                                    )))

        # model.add(Conv1D(filters=1,
        #                  kernel_size=16,
        #                  strides=1,
        #                  padding='causal',
        #                  input_shape=(x.shape[1], x.shape[2])))
        # model.add(Flatten())

        # model.add(Dropout(0.2))
        model.add(Dense(num_tokens))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.1, decay=0.1)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        model.summary()

        def on_epoch_end(epoch, logs):
            print()
            print('----- Generating text after Epoch: %d' % epoch)

            tweet_index = random.randint(0, len(tweets))
            for diversity in [None, 0.2, 0.5, 1.0, 1.2]:
                print('----- diversity:{}'.format(diversity))

                sentence = tweets[tweet_index][:self.context_char_size]
                generated = sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(179 - self.context_char_size):
                    x_pred = np.zeros((1, self.context_char_size, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = self.sample(preds, diversity)
                    next_char = indices_char[next_index]

                    if next_char == self._eos_char:
                        break

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        model.fit(prefixes, next_tokens,
                  verbose=1,
                  batch_size=128,
                  epochs=60,
                  callbacks=[
                      # print_callback,
                      # TensorBoard('tensorboard_logs', histogram_freq=1),
                      ModelCheckpoint('model.h5', verbose=1)])


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
