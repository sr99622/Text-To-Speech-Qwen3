Create a Python virtual environment, activate it, then run this script to install Qwen3-TTS. The program will run nvidia-smi to determine your CUDA version, then coordinate the CUDA version with a prebuilt binary for flash attention using https://github.com/mjun0812/flash-attention-prebuild-wheels. Please sponsor Junya Morioka if you are using the binaries. You can enter your github token using the environment variable GITHUB_TOKEN to speed up this process.

example uses

To preview the installation instructions
```
python qwen_tts_install.py preview
```

To get the installation instructions in json format
```
python qwen_tts_install.py resolve --json
```

To have the program do the actual installation, with a confirmation prompt
```
python qwen_tts_install.py install
```

To install without prompting
```
python qwen_tts_install.py install --yes
```
