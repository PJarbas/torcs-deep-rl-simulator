# 🏎️ TORCS Deep RL Simulator

<div align="center">

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![TORCS](https://img.shields.io/badge/TORCS-simulator-orange.svg)](http://torcs.sourceforge.net/)

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/torcs-racing/torcs-racing.github.io/master/img/torcs-banner.png" alt="TORCS Banner" width="600">
</p>

Train autonomous racing agents using Deep Reinforcement Learning in the TORCS (The Open Racing Car Simulator) environment! 🤖 🏁

## ✨ Overview

This project provides a complete environment for training AI racing agents using state-of-the-art reinforcement learning algorithms:

- 🔥 PPO (Proximal Policy Optimization)
- 🌟 SAC (Soft Actor-Critic)

The project uses modern Python features and best practices:
- 📝 Full type hints support
- 🏗️ Modular and extensible architecture
- 🔍 Clean code with comprehensive documentation
- 🧪 Ready for testing and experimentation

## 🛠️ Requirements

- 🐳 Docker and Docker Compose
- 🐍 Python 3.7+
- 🎮 NVIDIA GPU (recommended for faster training)

### Development Requirements

For development, you might also want to install:
- 🔍 mypy (for type checking)
- 📝 black (for code formatting)
- 🧪 pytest (for testing)

You can install them with:
```bash
pip install mypy black pytest
```

## 📦 Installation

1. 📥 Clone the repository:
```bash
git clone <repository-url>
cd torcs-deep-rl-simulator
```

2. Choose your preferred setup method:

### 🐳 Using Docker (Recommended)

The project includes a Makefile with helpful commands:

```bash
make build      # 🏗️ Build Docker images
make up         # 🚀 Start all services
make down       # 🛑 Stop all services
make train      # 🎯 Run training with PPO
make play       # 🎮 Run demo play script
make web        # 🌐 Start web interface
make clean      # 🧹 Clean up Docker resources
```

Alternatively, use Docker Compose directly:
```bash
docker-compose up --build
```

This command will launch both the TORCS simulator and training application! 🚀

### 💻 Local Installation

1. 🛠️ Create and setup the virtual environment:
```bash
./create_env.sh
```

The `create_env.sh` script will:
- 📦 Create a new virtual environment in `.venv`
- 🔄 Update pip to the latest version
- ⬇️ Install all required dependencies

2. ⚡ Activate the virtual environment:
```bash
source .venv/bin/activate  # 🐧 Linux/Mac
# or
.\.venv\Scripts\activate  # 🪟 Windows
```

3. 🎯 Run the local setup script:
```bash
./scripts/run_local.sh
```

## 🚀 Usage

### 🎯 Training an Agent

1. 📋 The project includes example configurations for PPO and SAC algorithms in `src/training/configs/`.

2. 🚀 To start training with PPO:
```bash
./examples/demo_train.sh
```

Or manually:
```bash
python src/training/train.py --config src/training/configs/ppo.yaml
```

3. 🌟 To use SAC instead:
```bash
python src/training/train.py --config src/training/configs/sac.yaml
```

### ⚙️ Configuration

Customize your training with YAML configuration files:

- 🎯 `src/training/configs/ppo.yaml`: PPO algorithm configuration
- 🌟 `src/training/configs/sac.yaml`: SAC algorithm configuration

Key parameters:
- 🤖 `algorithm`: Choose between "PPO" or "SAC"
- 👁️ `vision`: Enable/disable vision-based learning
- ⏱️ `total_timesteps`: Total number of training steps
- 🎛️ `agent_kwargs`: Algorithm-specific parameters

### 🌐 Web Interface

Monitor your training progress through our web interface:

1. 🚀 Automatically starts with Docker
2. 🔗 Access at: http://localhost:8000

## 📁 Project Structure

```
src/
├── agents/             # 🤖 RL agent implementations
├── envs/              # 🌍 TORCS environment wrapper
├── training/          # 🎯 Training scripts and configs
├── utils/             # 🛠️ Utility functions
├── web/               # 🌐 Web interface
└── examples/          # 📚 Example scripts
```

## 👩‍💻 Development

### Type Checking

The project uses type hints throughout the codebase. To check types:

```bash
mypy src/
```

### Code Style

To format the code according to the project's style:

```bash
black src/
```

### Testing

To run the tests:

```bash
pytest tests/
```

### Architecture

The project follows a modular architecture:

- 🎯 **Agents**: Implement the `BaseAgent` interface for new algorithms
- 🌍 **Environment**: TORCS wrapper implementing the Gymnasium interface
- 🧠 **Features**: Custom CNN feature extractor for visual and state inputs
- 🎛️ **Configuration**: YAML-based configuration for easy experimentation

## 🤝 Contributing

We welcome contributions! Feel free to:
- 🐛 Open issues
- 🚀 Submit pull requests
- 💡 Share your ideas
- ⭐ Star the project if you like it!

### Contributing Guidelines

1. Fork the repository
2. Create a new branch for your feature
3. Add type hints to new code
4. Ensure all tests pass
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

