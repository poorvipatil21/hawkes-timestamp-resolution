# ECE 621 Lab 4
**Name:** Poorvi Patil
**Hardware:** Google Colab A100

### Technical Notes
- DQN: Used CnnPolicy with a 100k buffer.
- PPO: Used parallelized environments (n_envs=8).
- VLM: Used Qwen2-VL-2B via Transformers.

### Reflection Questions
(a) Most impactful change: Increasing parallel environments for PPO to stabilize updates.
(b) VLM vs Random: The VLM showed basic awareness but lacked the precision of the RL agents.
(c) Resolution: RL agents succeeded despite lower resolution because they are specialized for the task through training, whereas the VLM is a generalist.