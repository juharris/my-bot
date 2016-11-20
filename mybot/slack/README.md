# Slack

Replies to Slack messages for you.

## Setup

1. Use Python 2.7 (as per the [docs](https://slackapi.github.io/python-slackclient/)).
2. `pip install --requirement requirements.txt`
3. Get a Slack API token. See [here](https://api.slack.com/docs/oauth-test-tokens).
4. Set `slack API token` in config.yaml using the token from the previous step.
5. Set `user id` in config.yaml with your user ID.  This avoids replying to your own messages. We'll try to determine this automatically in the future and remove this step.
6. (Optional) Add to `reply rules` in config.yaml.

## Deployment

If you want to store the configuration somewhere else, then you can change the path to the configuration file using the `MYBOT_SLACK_CONFIG_PATH` environment variable.

### Simple Deployment
```sh
python bot.py
```

### Docker Deployment
Build the Docker container:
```sh
docker build -t mybot .
```

Run the container:
```sh
docker run -it -d -v ${PWD}:/usr/src/app --name mybot mybot
```

To run the container with a specific configuration path:
```sh
docker run -it -d -v ${PWD}:/usr/src/app -e "MYBOT_SLACK_CONFIG_PATH=/path/to/config.yaml" --name mybot mybot
```
