# Serious Experiments
🚀 **Welcome to Serious Experiments: Unleashing the Quantum Leap in Machine Learning!** 🚀

Dive into the future of innovation with AwesomeProject, the unrivaled champion of cutting-edge machine learning in the uncharted realms of the quantum universe. 🌌🛸 Brace yourself for a mind-bending journey that defies the boundaries of conventionality and propels us into the pantheon of the avant-garde.

**🔮 Mastering Quantum Enigmas: Our Quest for the Ultimate Insight 🧠💥**
With a team of mad scientists, quantum sorcerers, and data whisperers, AwesomeProject fearlessly embarks on deciphering the secrets of the quantum multiverse. Witness the transformation of mere data points into predictive marvels that rewrite the playbook of machine learning. Our ethereal algorithms ride the wavelengths of uncertainty, bending reality to deliver the state-of-the-art results that mere mortals could only dream of.

**🌌 A Symphony of Chaos and Clarity: The MyCoolModel Advantage 🎶🌪️**
Immerse yourself in the symphony of chaos and clarity that our MyCoolModel orchestrates. Our model defies traditional norms, combining neural networks, quantum entanglement, and a dash of cosmic stardust to achieve results that will have you redefining your expectations of reality. Be it image recognition, language generation, or financial forecasting, MyCoolModel reigns supreme in the pursuit of the unknown.

**⚡️ Quantum-Powered Toolkit: Equip Yourself for the Extraordinary 🛠️🔑**
AwesomeProject isn't just about unveiling the universe's secrets; it's also about empowerment. Dive into our quantum-powered toolkit that equips you with the instruments needed to tackle the most intricate challenges in data science. Quantum data preprocessing, parallel universe hyperparameter tuning, and time-traveling gradient boosting – these are just a few tricks up our sleeves to catapult your skills to infinity and beyond.

**🚄 Fast-Track Your Learning Curve: Leap into Tomorrow, Today! 🌟🚀**
The future of machine learning has arrived, and AwesomeProject holds the keys to unlock it. From novices to quantum gurus, we invite all brave souls to step onto the express train to the next dimension of innovation. Unearth the secrets, master the art, and become a trailblazer in the evolution of technology. With AwesomeProject, tomorrow is already within your grasp.

**🔗 Connect with the Quantum Collective: Join the Revolution! 🌐🌈**
Join us in our pursuit of the unknown, and be part of the Quantum Collective that is reshaping the world of machine learning. Collaborate with quantum pioneers, share your breakthroughs, and push the boundaries of what's possible. Connect, collaborate, and catapult humanity into a future that's only limited by our imagination.

Prepare to be dazzled, awed, and utterly transformed. Welcome to AwesomeProject – where we don't just predict the future; we create it. 🌠🚀

## Koala is Watching
                 |       :     . |  
                 | '  :      '   |
                 |  .  |   '  |  |
       .--._ _...:.._ _.--. ,  ' |
      (  ,  `        `  ,  )   . |
       '-/              \-'  |   |
         |  o   /\   o  |       :|
         \     _\/_     / :  '   |
         /'._   ^^   _.;___      |
       /`    `""""""`      `\=   |
     /`                     /=  .|
    ;             '--,-----'=    |
    |                 `\  |    . |
    \                   \___ :   |
    /'.                     `\=  |
    \_/`--......_            /=  |
                |`-.        /= : |
                | : `-.__ /` .   |
                |    .   ` |    '|
                |  .  : `   . |  |

## Installation

### Conda

```bash
conda env create -f environment.yml
```

### Local

```bash
pip install .
```

or

```bash
pip install -r requirements.txt
```


## Reproduce Our Results
```
python train_sb.py --algorithm PPO \
                   --num_envs 10 \
                   --env_name CartPole-v1 \
                   --learning_rate 0.001 \
                   --batch_size 64 \
                   --policy_kwargs "dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64])" \
                   --max_steps 200000 \
                   --eval_freq 1000 \
                   --save_freq 50000 \
                   --n_eval_episodes 10 \
                   --log_interval 100 \
                   --reward_threshold 400
```