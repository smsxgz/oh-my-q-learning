Oh My Q-Learning!
----
                      /^ ^\
                     / 0 0 \
                     V\ Y /V
                      / - \
                     /    |
                    V__) ||

Ascii Art is from [ascii-code.com](https://www.ascii-code.com/ascii-art/animals/dogs.php).


Here is my implementation of the Q-learning algorithms by tensorflow.
* DQN
* Double DQN
* [Distributional DQN](https://arxiv.org/abs/1707.06887)
* [Dueling Network](https://arxiv.org/pdf/1511.06581)
* Combine Dueling network and distributonal DQN like [Rainbow](https://arxiv.org/pdf/1710.02298)

----
My thanks for [Denny Britz](https://github.com/dennybritz/reinforcement-learning) and Min RK.

lib/atari_wrapper.py is from [baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
lib/ale_wrapper.py is from [Wenhao Yang](https://github.com/yangwenh).

----
smsxgz@gmail.com

----
2017/10/31
Now you can run the code by:
```bash
python train/gym_atari_train.py -g BeamRider-v4 -p distribution
```
or:
```bash
python train/simple_train.py -g CartPole-v0 -p distribution --num_agent 1 --num_worker 1
```

----
2017/11/1
Update ALE Wrapper.
```bash
python train/ale_train.py -g beam_rider -p ale
```

----
TODO:
1. Dueling network.
1. Combine Dueling network and distributonal DQN like Rainbow.
1. More experiments.
