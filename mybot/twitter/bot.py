import itertools
import json
import logging
import os
import sqlite3
import time
from typing import List, Iterable

import keras.backend as K
import numpy as np
import twitter
from injector import inject, Injector, ProviderOf, singleton
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from keras.layers import Activation, Bidirectional, Dense, Embedding, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from mybot.twitter.constants import Configuration
from mybot.twitter.db_module import DbModule
from mybot.twitter.module import MyTwitterBotModule


@singleton
class MyTwitterBot(object):
    @inject
    def __init__(self, config: Configuration,
                 db_provider: ProviderOf[sqlite3.Connection],
                 logger: logging.Logger):
        self._config = config
        self._db_provider = db_provider
        self._logger = logger

        twitter_api_config = self._config['Twitter']
        self._twitter_api = twitter.Api(
            sleep_on_rate_limit=True,
            consumer_key=twitter_api_config['consumer key'],
            consumer_secret=twitter_api_config['consumer secret'],
            access_token_key=twitter_api_config['access token key'],
            access_token_secret=twitter_api_config['access token secret'])

        self._max_num_context_tokens = 10
        self._begin_token = '--bos--'
        self._eos_token = '--eos--'
        self._pad_token = '--pad--'
        self._pad_index = 0

        self._embedding_dim = 300
        self._batch_size = 192

        self._max_tweet_chars = 180

    def collect_friend_tweets(self, screen_name):
        """
        Get tweets of the people `screen_name` is following.
        :param screen_name: A Twitter username.
        """
        following_ids = self._twitter_api.GetFriendIDs(screen_name=screen_name)
        self._logger.info("Found %d friends", len(following_ids))
        result = []
        for friend_id in following_ids:
            self.collect_tweets(user_id=friend_id)
        return result

    def collect_tweets(self, user_id: int = None, screen_name: str = None):
        user = self._twitter_api.GetUser(user_id=user_id, screen_name=screen_name)
        user_id = user.id
        screen_name = user.screen_name

        db = None
        cursor = None

        try:
            db = self._db_provider.get()
            cursor: sqlite3.Cursor = db.cursor()

            max_id = None

            cursor.execute('SELECT * FROM tweeter WHERE user_id = ? ',
                           (user_id,))
            users = cursor.fetchall()
            assert len(users) <= 1
            if len(users) == 0:
                self._logger.info("Adding new user to DB for @%s.", screen_name)
                cursor.execute('INSERT INTO tweeter VALUES (?, ?, ?) ',
                               (user_id, screen_name, max_id))
            else:
                max_id = users[0][2]

            desc = "Getting tweets from @{} with max ID {}".format(screen_name, max_id)
            with tqdm(desc=desc,
                      unit_scale=True, mininterval=2, unit=" tweets") as pbar:
                while True:
                    if max_id is not None:
                        # Subtract 1 since getting tweets is inclusive.
                        max_id -= 1
                    tweets: List[twitter.Status] = self._twitter_api.GetUserTimeline(
                        user_id=user_id,
                        screen_name=screen_name,
                        max_id=max_id)
                    if len(tweets) == 0:
                        break
                    if max_id is None:
                        max_id = tweets[0].id

                    tweets_to_save = []
                    for tweet in tweets:
                        max_id = min(max_id, tweet.id)
                        text = tweet.text.strip()
                        if len(text) > 2:
                            tweets_to_save.append((user_id, text))

                    cursor.executemany('INSERT INTO tweet VALUES (?, ?)', tweets_to_save)
                    cursor.execute('UPDATE tweeter SET oldest_tweet_id = ? '
                                   '  WHERE user_id = ?',
                                   (max_id, user_id))

                    db.commit()
                    pbar.update(n=len(tweets_to_save))
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

    def get_all_stored_tweets(self):
        db = None
        cursor = None

        try:
            db = self._db_provider.get()
            cursor: sqlite3.Cursor = db.cursor()
            for tweet in cursor.execute('SELECT tweet_text FROM tweet'):
                yield tweet[0]
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

    def get_stored_tweets(self, user_id: int):
        db = None
        cursor = None

        try:
            db = self._db_provider.get()
            cursor: sqlite3.Cursor = db.cursor()
            for tweet in cursor.execute('SELECT tweet_text FROM tweet'
                                        '  WHERE user_id = ?',
                                        (user_id,)):
                yield tweet[0]
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

    def get_tweets_for_pre_training(self) -> Iterable[str]:
        result = set(self.get_all_stored_tweets())
        result = map(self.pre_process, result)
        result = map(lambda t: "{} {} {}".format(self._begin_token, t, self._eos_token), result)
        return result

    def get_tweets_for_training(self, user_id: int) -> Iterable[str]:
        result = set(self.get_stored_tweets(user_id))
        result = map(self.pre_process, result)
        result = map(lambda t: "{} {} {}".format(self._begin_token, t, self._eos_token), result)
        return result

    def run(self):
        model = load_model('model.h5')
        with open('id2token.json', encoding='utf-8') as f:
            id2token = json.load(f)
        id2token = {int(index): token for (index, token) in id2token.items()}
        token2id = {token: index for (index, token) in id2token.items()}

        while True:
            sentence = [token2id[self._begin_token]]
            generated = []

            for _ in range(40):
                x_pred = pad_sequences([sentence], maxlen=self._max_num_context_tokens)
                preds = model.predict(x_pred)[0]
                next_index = self.sample(preds, temperature=None)
                next_token = id2token.get(next_index)

                if next_token == self._eos_token:
                    break

                if next_token is not None:
                    sentence.append(next_index)
                    generated.append(next_token)

            generated = " ".join(generated)
            # self._twitter_api.PostUpdate(generated)
            self._logger.info("Generated: \"%s\"",
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

    def train(self, screen_name: str):
        user_id = self._twitter_api.GetUser(screen_name=screen_name).id

        tweets = list(self.get_tweets_for_pre_training())

        self._logger.info("Setting up tokenizer.")
        # TODO Keep punct as separate tokens.
        tokenizer = Tokenizer(filters='!"$%&()*+,./:;<=>?[\\]^`{|}~\t\r\n')
        tokenizer.fit_on_texts(tweets)

        token_index = tokenizer.word_index
        assert self._pad_index == 0
        token_index[self._pad_token] = self._pad_index
        self._logger.info("Token index size = %d", len(token_index))
        id2token = {index: token for (token, index) in token_index.items()}
        with open('id2token.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(id2token, separators=(',', ':')))

        sequences = tokenizer.texts_to_sequences_generator(tweets)
        steps_per_epoch = self._get_num_training_steps(sequences)
        sequences = tokenizer.texts_to_sequences_generator(tweets)
        training_data_generator = self._get_training_data(token_index, sequences)

        self._logger.info("Setting up the model.")
        model = self._create_model(token_index)

        def on_epoch_end(epoch, logs):
            print()
            print("----- Generating text after epoch: {}".format(epoch))

            for diversity in ['argmax', None, 0.2, 0.5, 1.0, 1.2]:
                print("----- diversity: {}".format(diversity))

                sentence = [token_index[self._begin_token]]
                print("----- Generating with seed: \"{}\"".format(self._begin_token))
                for i in range(40):
                    x_pred = pad_sequences([sentence], maxlen=self._max_num_context_tokens)
                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = self.sample(preds, diversity)
                    next_token = id2token.get(next_index)

                    if next_token == self._eos_token:
                        break

                    if next_token is not None:
                        sentence.append(next_index)
                        try:
                            print(next_token, end=" ")
                        except:
                            print("ERR", end=" ")
                print()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        t = int(time.time())
        os.makedirs('tensorboard_logs', exist_ok=True)
        callbacks = [
            TensorBoard('tensorboard_logs/pre-train-{}-{}'.format(screen_name, t), histogram_freq=1),
            print_callback,
            ModelCheckpoint('pre-trained-{}-model.h5'.format(screen_name), verbose=1)]

        # Mainly just do validation so that we can TensorBoard.
        model.fit_generator(itertools.islice(training_data_generator, 1, None),
                            steps_per_epoch=steps_per_epoch - 1,
                            validation_data=itertools.islice(training_data_generator, 1),
                            validation_steps=1,
                            verbose=1,
                            epochs=30,
                            callbacks=callbacks)

        self._logger.info("Done pre-training. Training on specific data.")

        tweets = list(self.get_tweets_for_training(user_id))
        sequences = tokenizer.texts_to_sequences_generator(tweets)
        steps_per_epoch = self._get_num_training_steps(sequences)
        sequences = tokenizer.texts_to_sequences_generator(tweets)
        training_data_generator = self._get_training_data(token_index, sequences)
        callbacks = [
            TensorBoard('tensorboard_logs/train-{}-{}'.format(screen_name, t), histogram_freq=1),
            print_callback,
            ModelCheckpoint('model-{}.h5'.format(screen_name), verbose=1)]

        model.fit_generator(itertools.islice(training_data_generator, 1, None),
                            steps_per_epoch=steps_per_epoch - 1,
                            validation_data=itertools.islice(training_data_generator, 1),
                            validation_steps=1,
                            verbose=1,
                            epochs=60,
                            callbacks=callbacks)

    def _create_model(self, token_index):
        embeddings_matrix = np.zeros((len(token_index), self._embedding_dim))
        assert self._pad_index == 0
        # Pad vector is all 0s so consider it already loaded in the first row.
        loaded_indices = {self._pad_index}
        with open(os.path.expanduser(self._config['embeddings path']), encoding='utf-8') as f:
            for line in tqdm(f,
                             desc="Loading embeddings",
                             unit_scale=True, mininterval=2, unit=" tokens"):
                values = line.split(maxsplit=1)
                token = values[0]
                index = token_index.get(token)
                if index is not None:
                    loaded_indices.add(index)
                    vec = values[1].split()
                    assert len(vec) == self._embedding_dim
                    vec = np.asarray(vec, dtype=np.float32)
                    embeddings_matrix[index] = vec
        self._logger.info("Adding vectors for %d OOV tokens.",
                          embeddings_matrix.shape[0] - len(loaded_indices))

        # Handle tokens with no existing embeddings.
        for i in range(embeddings_matrix.shape[0]):
            if i not in loaded_indices:
                # Make a new vector.
                # FIXME Make new vec with same magnitude as others.
                # FIXME Make new vec far from others.
                embeddings_matrix[i] = np.random.uniform(low=-0.5, high=0.5, size=(self._embedding_dim,)).astype(
                    np.float32)
        del loaded_indices

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
        model.add(Activation('softmax'))
        model.add(Dense(len(token_index)))
        optimizer = RMSprop(lr=0.01, decay=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        model.summary()
        return model

    def _get_num_training_steps(self, sequences):
        # Coupled with _get_training_data.
        result = 0

        step = 1
        for tweet in sequences:
            result += len(range(1, len(tweet), step))

        result = int(result / self._batch_size)
        return result

    def _get_training_data(self, token_index, sequences):
        # Coupled with _get_num_training_steps.
        step = 1
        prefixes, y = [], []

        for tweet in tqdm(sequences,
                          desc="Building training data",
                          unit_scale=True, mininterval=2, unit=" tweets"):
            # Get sequences with just the first token,
            # and window through until we have sequences up to before the last token.
            for next_token_index in range(1, len(tweet), step):
                prefixes.append(tweet[max(0, next_token_index - b._max_num_context_tokens): next_token_index])
                token_vec = np.zeros(len(token_index), dtype=np.int32)
                token_vec[tweet[next_token_index]] = 1
                y.append(token_vec)

                if len(prefixes) == self._batch_size:
                    prefixes = pad_sequences(prefixes, maxlen=self._max_num_context_tokens)
                    y = np.array(y)
                    yield (prefixes, y)
                    prefixes, y = [], []


if __name__ == '__main__':
    injector = Injector([
        DbModule,
        MyTwitterBotModule,
    ])
    b: MyTwitterBot = injector.get(MyTwitterBot)
    # b.collect_friend_tweets(screen_name='chelsdelaney11')
    # b.collect_tweets(screen_name='chelsdelaney11')
    b.train(screen_name='chelsdelaney11')
    # b.run()
