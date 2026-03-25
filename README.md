<h3>Installation</h3>

Required Enviroment: Linux equipped with NVIDIA GPU running CUDA, nvidia-smi must work. You will need git and ffmpeg.

<b>What the script does</b>

This script will install [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with Flash Attention enabled for maximum performance. The script will run nvidia-smi to determine your CUDA version, then coordinate the proper version of [PyTorch](https://pytorch.org/get-started/locally/) with a prebuilt binary for flash attention built by Junya Morioka [Flash Attention 2](https://github.com/mjun0812/flash-attention-prebuild-wheels).


<b>To begin installation</b>

Create a Python virtual environment and activate, then run the following commands. The installer will test your environment and show the proposed installation commands before requesting confirmation.


```
git clone https://github.com/sr99622/Text-To-Speech-Qwen3
cd Text-To-Speech-Qwen3
pip install -r requirements.txt
python qwen_tts_install.py install
```


<details><summary>Other Installer Configurations</summary>

To preview the installation instructions
```
python qwen-tts-install.py preview
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
python qwen-tts-install.py install --yes
```

</details>

---

<h3>Voice Design GUI</h3>

Create a new voice with a prompt.

```
python voice_design/voice_design_gui.py
```

&nbsp;

<image src="assets/voice_design_gui.png">

&nbsp;

Qwen3-TTS has the ability to create voices using a prompt describing the voice. It has been observed that there is quite a bit a variability in the quality of the output that can have a dramatic effect on the tone of the speech. It is helpful to make several runs of the speech generation so that there are a number of candidates for the final selection of the audio clip.

The program takes a voice quality prompt that describes the qualities the type of voice desired and a script to be read by the voice. Then a batch process is run that will produce a number of candidates for the voice, which can be played back from inside the app. File names can be changed from inside the app to make it easier to find them later on. Prompts are saved along with the audio files they generate so there is context for the generation. Use the Open Batch Folder button to access the history of generations, and reload them for review.

---

<h3>Voice Clone GUI</h3>

Use an existing voice to read a script.

```
python voice_clone/voice_clone_gui.py
```
&nbsp;

<image src="assets/voice_clone_gui.png">

