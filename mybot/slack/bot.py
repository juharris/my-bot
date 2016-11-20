from __future__ import print_function, unicode_literals

import io
import json
import logging
import os
import random
import re
import time
from collections import namedtuple

import yaml
from slackclient import SlackClient


class MySlackBot(object):
    def __init__(self):
        config_path = os.getenv('MYBOT_SLACK_CONFIG_PATH')
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')

        print("Loading config from " + config_path)
        with io.open(config_path, 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)

        self._logger = logging.getLogger('mybot.slack')
        log_level = config.get('log level', logging.WARN)
        self._logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s\n%(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        slack_token = config["slack API token"]
        self._client = SlackClient(slack_token)
        del slack_token

        # Get current user's ID so that they don't automatically reply to their own messages.
        # TODO Determine is automatically: might need to use OAuth.
        self._current_user_id = config.get('user id')

        self._reply_rules = []
        ReplyRule = namedtuple('ReplyRule', ['pattern', 'replies'])
        for rule in config.get('reply rules', []):
            self._reply_rules.append(ReplyRule(re.compile(rule['pattern']), rule['replies']))

        self._check_delay = config.get('check delay', 3)

    def _get_reply(self, received_msg):
        result = None
        for rule in self._reply_rules:
            m = rule.pattern.match(received_msg)
            if m:
                result = random.choice(rule.replies)
                break
        return result

    def handle_messages(self):
        if self._client.rtm_connect():
            self._logger.info("Waiting for messages.")
            while True:
                try:
                    rtm_results = self._client.rtm_read()
                except Exception as e:
                    self._logger.exception(e)
                    continue
                for event in rtm_results:
                    try:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug("Event: \"%s\"", json.dumps(event, indent=2))
                        if event.get('type') == 'message' and event.get('user') != self._current_user_id:
                            channel = event.get('channel', '')
                            if channel.startswith('D'):
                                received_msg = event.get('text')
                                reply = self._get_reply(received_msg)
                                if reply:
                                    self._logger.info("Replying: \"%s\" in %s", reply, channel)
                                    self._client.rtm_send_message(channel, reply)
                    except Exception as e:
                        self._logger.exception(e)
                time.sleep(self._check_delay)
        else:
            raise Exception("Connection failed. Maybe your token is invalid?")


def main():
    my_bot = MySlackBot()
    my_bot.handle_messages()


if __name__ == '__main__':
    main()
