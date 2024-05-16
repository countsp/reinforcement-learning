<p align="center"><img width="90%" src="images/Reinforcement-Learning.png"></p>

--------------------------------------------------------------------------------

> Minimal and clean examples of reinforcement learning algorithms presented by [RLCode](https://rlcode.github.io) team. [[한국어]](https://github.com/rlcode/reinforcement-learning-kr)
>
> Maintainers - [Woongwon](https://github.com/dnddnjs), [Youngmoo](https://github.com/zzing0907), [Hyeokreal](https://github.com/Hyeokreal), [Uiryeong](https://github.com/wooridle), [Keon](https://github.com/keon)

From the basics to deep reinforcement learning, this repo provides easy-to-read code examples. One file for each algorithm.
Please feel free to create a [Pull Request](https://github.com/rlcode/reinforcement-learning/pulls), or open an [issue](https://github.com/rlcode/reinforcement-learning/issues)!

## Dependencies
1. Python 3.5
2. Tensorflow 1.0.0
3. Keras
4. numpy
5. pandas
6. matplot
7. pillow
8. Skimage
9. h5py

### Install Requirements
```
pip install -r requirements.txt
```

## Table of Contents

**Grid World** - Mastering the basics of reinforcement learning in the simplified world called "Grid World"

- [Policy Iteration](./1-grid-world/1-policy-iteration)
- [Value Iteration](./1-grid-world/2-value-iteration)
- [Monte Carlo](./1-grid-world/3-monte-carlo)
- [SARSA](./1-grid-world/4-sarsa)
- [Q-Learning](./1-grid-world/5-q-learning)
- [Deep SARSA](./1-grid-world/6-deep-sarsa)
- [REINFORCE](./1-grid-world/7-reinforce)

**CartPole** - Applying deep reinforcement learning on basic Cartpole game.

- [Deep Q Network](./2-cartpole/1-dqn)
- [Double Deep Q Network](./2-cartpole/2-double-dqn)
- [Policy Gradient](./2-cartpole/3-reinforce)
- [Actor Critic (A2C)](./2-cartpole/4-actor-critic)
- [Asynchronous Advantage Actor Critic (A3C)](./2-cartpole/5-a3c)

**Atari** - Mastering Atari games with Deep Reinforcement Learning

- **Breakout** - [DQN](./3-atari/1-breakout/breakout_dqn.py), [DDQN](./3-atari/1-breakout/breakout_ddqn.py) [Dueling DDQN](./3-atari/1-breakout/breakout_ddqn.py) [A3C](./3-atari/1-breakout/breakout_a3c.py)
- **Pong** - [Policy Gradient](./3-atari/2-pong/pong_reinforce.py)

**OpenAI GYM** - [WIP]

- Mountain Car - [DQN](./4-gym/1-mountaincar)


---

### Q学习的基本概念

1. **状态（State）**：
   状态是智能体在某一时刻所处的情况。例如，在迷宫中，状态可以是智能体所在的位置。

2. **动作（Action）**：
   动作是智能体可以采取的行为，例如在迷宫中，动作可以是向上、向下、向左或向右移动。

3. **奖励（Reward）**：
   奖励是智能体在采取某个动作后得到的反馈。例如，如果智能体找到出口，可以获得一个正的奖励，如果撞到墙壁，可能会得到负的奖励。

4. **Q值（Q-value）**：
   Q值是一个数字，用来表示在特定状态下采取某个动作的好坏。智能体通过学习来更新这些Q值，从而知道在不同的状态下，哪种动作是最好的。

### Q学习的工作原理

Q学习的目标是找到一种策略，使得智能体在任何状态下都能选择最优的动作。具体来说，Q学习通过以下步骤来实现这个目标：

1. **初始化**：
   智能体首先会创建一个Q值表，这个表格记录了在所有可能状态下采取所有可能动作的Q值。初始时，这些Q值通常设为零。

2. **选择动作**：
   在每个状态下，智能体会根据当前的Q值表来选择一个动作。选择动作的方法包括：
   - **探索（Exploration）**：随机选择一个动作，目的是探索新的可能性。
   - **利用（Exploitation）**：选择当前Q值最高的动作，目的是利用已知信息做出最佳决策。

3. **执行动作并获得奖励**：
   智能体在选择并执行一个动作后，会进入一个新的状态，并获得一个奖励。

4. **更新Q值**：
   使用贝尔曼方程更新Q值，公式如下：
   
   $$
   Q(\text{当前状态}, \text{动作}) = Q(\text{当前状态}, \text{动作}) + \text{学习率} \times (\text{奖励} + \text{折扣因子} \times \text{下一状态的最大Q值} - Q(\text{当前状态}, \text{动作}))
 $$
   
   这个公式的意思是：新的Q值等于旧的Q值加上一个调整值，这个调整值反映了智能体从新的经验中学到的内容。

6. **重复**：
   智能体不断重复上述过程，随着时间的推移，Q值表会越来越精确，最终可以帮助智能体在任何状态下都能选择最优的动作。

### 总结

简单来说，Q学习是一种帮助计算机在复杂环境中找到最佳决策的方法。它通过不断尝试不同的动作、获取奖励和更新Q值表，最终学会在每种状态下做出最好的选择。

### example

所以回到之前的流程，根据Q表的估计，因为在s1中，a2的值比较大，通过之前的决策方法，我们在s1采取了a2，并到达s2，这是我们开始更新用于决策的Q表，接着我们并没有在实际中采取任何行为，而是再想象自己在s2上采取了每种行为，分别看看两种行为哪一个的Q值大，比如说Q(s2,a2)的值比Q(s2,a1)的大，所以我们把大的Q(s2,a2)乘上一个衰减值γ（比如是0.9）并加上到达s2时所获取的奖励R（这里我们还没有获取到棒棒糖，所以奖励为0）因为会获取实实在在的奖励R，我们将这个作为我现实中Q(s1,a2)的值，但是我们之前是根据Q表估计Q(s1,a2)的值。所以有了现实和估计值，我们就能更新Q(s1,a2)，根据估计与现实的差距，将这个差距乘以一个学习效率α累加上老的Q(s1,a2)的值变成新的值。但时刻记住，我们虽然用maxQ(s2)估算了一下s2的状态，但还没有在s2上做出任何的行为，s2的行为决策要等到更新完了以后再重新另外做。这就是off-policy的Q-Learning是如何决策和学习优化决策的过程。

![image](https://github.com/countsp/reinforcement-learning/assets/102967883/1478f7cb-d8d6-41c9-87c3-3351418c7fb6)

![image](https://github.com/countsp/reinforcement-learning/assets/102967883/89d5b9ce-0da5-45e0-ba46-5436ba516e53)

上图概括了之前所有的内容。这也是Q-Learning的算法，每次更新我们都用到了Q现实和Q估计，而且Q-Learning的迷人之处就是在Q(s1,a2)现实中，也包含了一个Q(s2)的最大估计值，将对下一步的衰减的最大估计和当前所得到的奖励当成这一步的现实。最后描述一下这套算法中一些参数的意义。ε-greedy是用在决策上的一种策略，比如ε=0.9时，就说明有90%的情况我会按照Q表的最优值选择行为，10%的时间使用随机选行为。α是学习率，来决定这次的误差有多少是要被学习的，α是一个小于1的数。γ是对未来reward的衰减值。我们可以这样想：

![image](https://github.com/countsp/reinforcement-learning/assets/102967883/537886df-9fcb-4849-b3e4-093ab5eac70d)

重写一下Q(s1)的公式，将Q(s2)拆开，因为Q(s2)可以像Q(s1)一样，是关于Q(s3)的，所以可以写成这样，然后以此类推，不停地这样写下去，最后就能携程这样，可以看出Q(s1)是有关于之后所有的奖励，但这些奖励正在衰减，离s1越远的状态衰减越严重。可以想象Q-Learning的机器人天生近视眼，γ=1时，机器人有了一副合适的眼镜，在s1看到的Q是未来没有任何衰变的奖励，也就是机器人能清清楚楚看到之后所有步的全部价值，但是当γ=0，近视机器人没了眼镜，只能摸到眼前的reward，同样也就只在乎最近的大奖励，如果γ从0变到1，眼镜的度数由浅变深，对远处的价值看得越清楚，所以机器人渐渐变得有远见，不仅仅只看眼前的利益，也为自己的未来着。

Q学习（Q-learning）是一种用于帮助计算机做出决策的算法。这种算法广泛应用于强化学习中，可以帮助智能体（例如机器人或计算机程序）在复杂环境中找到最佳的行为策略。
