# How to run

1. Install SMARTS following the instructions in `./SMARTS` folder

2. Download the original data from <https://jbox.sjtu.edu.cn/v/link/view/75a0931a222347e1ba2e0441407f4a1f> and place it under `./ngsim` folder

3. Build NGSIM scenario with `scl build --clean ./ngsim`

4. Generate expert demonstrations with `python example_expert_generation.py`

5. Test rollout with `python example_rollout.py`

# 项目说明

## 项目实现
我们实现了使用GAIL+TD3算法来实现驾驶决策任务，同时并使用Behavior Clone算法先预训练TD3模型，使得生成出来的决策分布和专家分布相近，减缓分布差距过大导致的梯度消失问题。此外还引入梯度惩罚和梯度裁剪来使得训练更加稳定。

## 效果展示

快车道效果
![image text](https://github.com/Merealtea/NGSIM_SMARTS/tree/main/display/图片一.gif)

慢车道效果
![image text](https://github.com/Merealtea/NGSIM_SMARTS/edit/main/display/图片二.gif)
