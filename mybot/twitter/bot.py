import io
import logging
import os
import random
import string
import sys
from collections import Counter
from operator import itemgetter
from typing import List

import numpy as np
import twitter
import yaml
from injector import inject, Injector, singleton
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import Activation, Bidirectional, Dense, GRU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from tqdm import tqdm


@singleton
class MyTwitterBot(object):
    @inject
    def __init__(self):
        config_path = os.getenv('MYBOT_TWITTER_CONFIG_PATH')
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')

        print("Loading config from `{}`.".format(config_path))
        with io.open(config_path, 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)

        self._logger = logging.getLogger('mybot.twitter')
        log_level = config.get('log level', logging.WARN)
        self._logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s:%(filename)s:%(funcName)s\n%(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        twitter_api_config = config['Twitter']
        self._twitter_api = twitter.Api(consumer_key=twitter_api_config['consumer key'],
                                        consumer_secret=twitter_api_config['consumer secret'],
                                        access_token_key=twitter_api_config['access token key'],
                                        access_token_secret=twitter_api_config['access token secret'])

        self.context_char_size = 40
        self._eos_char = '×'

    def get_tweets(self, screen_name):
        result = []
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
                    text = tweet.text.replace('\r', ' ').replace('\n', ' ').strip()
                    if len(text) > 2:
                        result.append(text)
                    pbar.update()
        self._logger.info("Found %d tweets.", len(result))

        seen = set()
        with open('tweets.csv', 'w', encoding='utf-8') as f:
            for text in result:
                if text not in seen:
                    f.write(text)
                    f.write('\n')
                    seen.add(text)

        return result

    def run(self):
        with open('tweets.csv', encoding='utf-8') as f:
            old_tweets = f.readlines()
        old_tweets = list(map(str.strip, old_tweets))

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
        with open('tweets.csv', encoding='utf-8') as f:
            text = f.read()
        tweets = text.split('\n')
        tweets = list(map(self.pre_process, tweets))

        chars = Counter(text)
        chars = filter(lambda c_count: c_count[1] >= 2, chars.most_common(n=26 + 10))
        chars = map(itemgetter(0), chars)
        chars = set(map(str.lower, chars))
        chars |= set(string.ascii_lowercase)
        chars |= set(string.digits)
        chars |= set(' \'".?!#:')
        chars = sorted(chars)
        assert self._eos_char not in chars, "Make a new EOS char."
        chars.append(self._eos_char)
        chars = sorted(chars)

        with open('chars.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(chars))

        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        step = 1
        sentences = []
        next_chars = []

        updated_tweets = []
        for tweet in tweets:
            # Filtered our OOV chars.
            tweet = ''.join([c for c in tweet if c in chars])
            updated_tweets.append(tweet)
        tweets = updated_tweets
        del updated_tweets

        for tweet in tweets:
            tweet += self._eos_char
            for i in range(0, len(tweet) - self.context_char_size, step):
                sentences.append(tweet[i: i + self.context_char_size])
                next_chars.append(tweet[i + self.context_char_size])
        print('nb sequences:', len(sentences))

        x = np.zeros((len(sentences), self.context_char_size, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        model = Sequential()

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
        model.add(Dense(len(chars)))
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

        model.fit(x, y,
                  verbose=1,
                  batch_size=128,
                  epochs=60,
                  callbacks=[print_callback,
                             # TensorBoard('tensorboard_logs', histogram_freq=1),
                             ModelCheckpoint('model.h5', verbose=1)])


def collect_tweets(screen_name):
    injector = Injector()
    b = injector.get(MyTwitterBot)
    b.get_tweets(screen_name)


if __name__ == '__main__':
    injector = Injector()
    b: MyTwitterBot = injector.get(MyTwitterBot)
    b.train()
    # b.run()
