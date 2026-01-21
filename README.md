## Walking H1

This project implements a custom **Proximal Policy Optimization (PPO)** algorithm from scratch to train locomotion policies for the **Unitree H1** humanoid robot. The environment is built on **IsaacLab / Isaac Sim**.

The project supports training, inference with checkpoints, and an interactive play mode with keyboard control for testing.

### Run PPO training:
```bash
./isaaclab.sh -p train.py 
```

Checkpoints will be saved under:
ckpts/

### Play (Inference + Keyboard Control)
Run inference from a saved checkpoint:
```bash
./isaaclab.sh -p play.py --ckpt ckpts/unitree_h1/model_final.pt
```

Keyboard controls:
- UP: move forward
- LEFT: turn left
- RIGHT: turn right
- DOWN: stop
- C: toggle camera view (third-person / perspective)
- ESC: exit current third-person view / clear selection
