# selfish-mining



Mingxun Zhou :



### Main File

**/simulator/rl/environment.py**

`SM_env(max_hidden_block, attacker_fraction, follower_fraction)` : RL environment, based on the model from  'Optimal Selfish Mining'. 

`max_hidden_block` : limitation of the MDP. 

`attacker_fraction` : $\alpha$ in the paper. Denotes the fraction of attacker's hash rate.

`follower_fraction` : $\gamma$ in the paper. Denotes the fraction of follower (against the total honest miners).

**states:** (a, b, status). 

a : length attacker's private fork

b : length of public fork

status :

​	`normal` : just normal. The attacker can perform everything he wants.

​	`forking` : the attacker's fork is competing with another fork. Both of them has b blocks, which means          the attacker still holds a-b private blocks.

​	`catch up` : the attacker mined a block, and now the length of private fork is the same as public fork, which means the attacker cannot perform `match`.

**action** :

`match ` : The attacker publish b blocks to compete with public fork. It will cause the public mining pool to separate and ` follower_fraction` of honest miners will follow the attacker's fork. If a>=b, then the next status will be `forking`. If a<b, then `match ` means the attacker abandon its private fork.

`override` : The attacker publish b+1 blocks to override public fork.

`wait` : The attacker just mines on his private fork.

The index for the action : `{'match':0, 'override':1, 'wait':2}`.

**reward** : 

Here the reward is *relative* reward. For every block in public chain acknowledged by both of the attacker and the honest miner will count the reward.  *(Some exploration space here...)*



`SM_env.reset()` : reset the environment and the statistical data, return the initiative state index.

`SM_env.step(idx, action)` : given the state index and action numbers, return the reward, next state index and reset flag.   *(very similar to OpenAI gym ...)* 

`SM_env.is_legal_move(idx, action)` : given a state and an action, judge if this action is legal in this situation.

`SM_env.observation_space_n` : the state space size

`SM_env.action_space_n` : the action space size

`SM_env.reward_fraction` : return the attacker's actual gain fraction in this run. 



**/simulator/rl/sim.py**

Table based RL.

Contains SARSA and Q-learning.

### Contributor

Yan Ji : yj348@cornell.edu

Mingxun Zhou : zhoumingxun@pku.edu.cn

