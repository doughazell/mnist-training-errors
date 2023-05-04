# mnist-training-errors
```
  e_t<-
 / /
s_f
```
### Introduction
Sequential DNN Supervised Learning with TF is based on: www.tensorflow.org/tutorials/quickstart/beginner
* Used in 'TFConfigTrain' to test training sets+epochs, DNN Layer config on digit recognition accuracy
* Used in 'TFConfigRL' to measure the efficacy of retraining inaccurate digits (rather than supervised learning training set)
  
  (It is planned to check (and retrain if necessary) a proportion of predictions via bitwise-AND of image to example image for each digit)

Categorical DQN Reinforcement Learning with TF is based on: https://github.com/tensorflow/agents/tree/master/docs/tutorials

### CartPole training with Categorical DQN
I changed the original tutorial video of multiple CartPole episodes at the end of training, to a one episode video every evaluation interval of 1000 training steps. This shows the increased time of a valid CartPole episode as the Categorical DQN is reinforcement trained.

The videos are stored in 'video/[Incremented Integer]/' directory along with the graph of “average return” (ie average step number of 10 episodes for the agent in an “evaluation gym”) for each interval.

### Install steps
```
$ brew install graphviz
$ brew install ffmpeg

$ pip install gspread==3.6.0
$ pip install oauth2client
$ pip install pydot
$ pip install 'imageio==2.4.0'
$ pip install pyvirtualdisplay
$ pip install pyglet
$ pip install --upgrade tensorflow
$ pip install --upgrade tf-agents
$ pip install pandas
$ pip install click

$ git clone https://github.com/doughazell/mnist-training-errors.git
```

Get 'client_secret.json' (originally got via https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html) from Google (and then adapt the spreadsheet code for your spreadsheet layout)

### Run 'tf-test'
```
$ cd mnist-training-errors
$ ./tf-test --help

  Usage: tf-test [OPTIONS] COMMAND [ARGS]...

    The Sequential DNN is trained with MNIST digits:

      a) high level with Supervised Learning

      b) 50% level prior to Reinforcement Learning

    Categorical DQN uses OpenAI Gym for CartPole physics

  Options:
    --help  Show this message and exit.

  Commands:
    cartpole  Train the TF Categorical DQN with Reinforcement Learning
    rl        Train the TF Sequential DNN with Reinforcement Learning
    train     Train the TF Sequential DNN with Supervised Learning
```
### Issues
Reverb only works on Linux (prob for commercial reasons) so this repo is platform independent (and done on my MacBook Pro) by using 'TFUniformReplayBuffer' (and not 'ReverbReplayBuffer' used in '1_dqn_tutorial.ipynb') 
* https://github.com/tensorflow/agents/issues/532#issuecomment-1230314133

