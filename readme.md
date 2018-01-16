夫祸患常积于忽微，而智勇多困于所溺。
-欧阳修《伶官传序》

----
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
* [Dueling Network](https://arxiv.org/pdf/151-[]06581)
* Combine Dueling network and distributonal DQN like [Rainbow](https://arxiv.org/pdf/1710.02298)

----
My thanks for [Denny Britz](https://github.com/dennybritz/reinforcement-learning) and Min RK.

scripts/atari_wrapper.py is from [baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
scripts/ale_wrapper.py is from [Wenhao Yang](https://github.com/yangwenh).

----
smsxgz@gmail.com

----
2018.1.16
经过仔细的思考，彻底抛弃了之前的想法，觉得应该写一个适合实验的框架，而不是一个复杂的似乎可以解决我所想的所有问题的框架。  
现在的框架在以前的我看来可能是比较丑陋的，但做实验比以前方便很多（主要是方便比较），这是一个小小的进步。  
我还会继续完善这个框架，现在暂时只有double DQN的实验，等下个月可能会逐渐补充其他的算法和Atari上的实验。
