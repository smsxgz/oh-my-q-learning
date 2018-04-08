2018.1.16  
经过仔细的思考，彻底抛弃了之前的想法，觉得应该写一个适合实验的框架，而不是一个复杂的似乎可以解决我所想的所有问题的框架。  
现在的框架在以前的我看来可能是比较丑陋的，但做实验比以前方便很多（主要是方便比较），这是一个小小的进步。  
我还会继续完善这个框架，现在暂时只有double DQN的实验，等下个月可能会逐渐补充其他的算法和Atari上的实验。

2018.2.17  
在CartPole-v1上的实验表明：selu和larger_batch_size可以带来更好的稳定性（到达500分后不会突然掉到100分的水平）

2018.3.5  
在Atari上的实验终于接近成功啦！详见 atari/config/double_dqn.v3

安装ffmpeg：  
```bash
conda install -c conda-forge ffmpeg
```

2018.3.7  
在Atari上的实验结果表明relu远比selu好，我有一个猜测：selu在全连接网络上表现会比relu好；在卷积网络上可能relu更好。  
PongNoFrameSkip-v4成功！

解决pip install 'gym[mujoco]'失败的方法：  
```bash
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
python setup.py install
```

2018.3.10  
pytorch版本的estimator终于调试成功啦，曲线也和valid padding版本的tensorflow一致啦!  
经过细致调研NoFrameskip-v4就是没有其他操作的ALE环境，所以不用特意改代码啦！  

2018.4.8  
经过一段时间的代码失效后，现在的v4终于运行正常了！  
改动：
1. 在每次reset后重设环境的随机数，使得eval变得可行
2. 在虚拟机上可运行，详见v4.tcp
3. 固定了epsilon，试验结果表明和之前的decay做法的训练曲线差别不大
3. 将discount属性加入Estimator，Estimator添加方法model.get_action(obs, epsilon)，
传入epsilon是为了将来如果想改回decay也非常方便  
