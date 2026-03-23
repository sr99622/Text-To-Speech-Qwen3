This program configures and installs Qwen3-TTS on Linux equipped with NVIDIA GPU running CUDA, nvidia-smi required.

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

<h3>Voice Design GUI</h3>

<image src="assets/voice_design_gui.png">

Qwen3-TTS has the ability to create voices using a prompt describing the voice. It has been observed that there is quite a bit a variability in the quality of the output that can have a dramatic effect on the tone of the speech. It is helpful to make several runs of the speech generation so that there are a number of candidates for the final selection of the audio clip.

The program takes a voice quality prompt that describes the qualities the type of voice desired and a script to be read by the voice. Then a batch process is run that will produce a number of candidates for the voice, which can be played back from inside the app. File names can be changed from inside the app to make it easier to find them later on. Prompts are saved along with the audio files they generate so there is context for the generation. Use the Open Batch Folder button to access the history of generations, and reload them for review.

In order to run this program, use the virtual environment created above for the installation and run

```
pip install pyqt6
```

Then start the program 

```
python voice_design/voice_design_gui.py
```
