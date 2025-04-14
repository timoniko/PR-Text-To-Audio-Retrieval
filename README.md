
This repository builds upon [salsa](https://github.com/OptimusPrimus/salsa) [[1]](#1) repository and contains the implementation to run and reproduce the results of two main successful experiments:
Text Aware Attention Pooling (TAP) and Overcaptioning. The experiments focused on stage 1 training using Clotho dataset. All runs were conducted on a single NVIDIA RTX4090


## Setting up the environment
To set up the environment, refer to [original repo](https://github.com/OptimusPrimus/salsa) and follow the instruction of the corresponding section,but with updated  ```environment.yaml``` file. 

## Baseline Run

To execute the baseline, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 with \
data_loader.batch_size=16 \
data_loader.batch_size_eval=16 \
audio_features.segment_length=10 \
audio_features.model=passt \
sentence_features.model=roberta-large \
rampdown_type=cosine \
max_epochs=20 \
rampdown_stop=15 \
warmup_length=1 \
rampdown_start=1 \
train_on=clothov2 \
seed=409194
```
The expected performance is: 

| map@10 |  R@1  |  R@5  | R@10  |
|:------:|:-----:|:-----:|:-----:|
| 30.16  | 18.96 | 44.93 | 59.44 |

Training and validation took 3 hours and 52 minutes.

## Text Aware Attention Pooling
Instead of mean pooling over audio frames, it is proposed in [[2]](#2) to allow model to attend to the most relevant frames to a provided text.

To execute the experiment, run the command:

```
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 with \
data_loader.batch_size=16 \
data_loader.batch_size_eval=16 \
audio_features.model=passt \
sentence_features.model=roberta-large \
rampdown_type=cosine \
max_epochs=20 \
rampdown_stop=15 \
warmup_length=1 \
rampdown_start=1 \
train_on=clothov2 \
seed=409194 \
audio_features.aggregate=TAP \
tap_hyperparams.p_dropout_weights=0.1 \
tap_hyperparams.p_dropout_proj=0.1 \
audio_features.segment_length=5 \
audio_features.hop_length=5
```

The expected performance is: 

| map@10 |  R@1  |  R@5  | R@10  |
|:------:|:-----:|:-----:|:-----:|
| 31.91  | 20.53 | 47.46 | 60.65 |

Training and validation took 1 hour and 40 minutes. 

## Data augmentation via generated captions.
Another experiment enables TAP along with LLM generated captions. It utilizes the Phi-4 language model which was run locally via [Ollama](https://ollama.com/).
Corresponding prompt and script are provided in ```experiments/generate_captions_phi4.py```. Generated captions are also available.

To execute the experiment, run the command:
```
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 with \
data_loader.batch_size=16 \
data_loader.batch_size_eval=16 \
audio_features.model=passt \
sentence_features.model=roberta-large \
rampdown_type=cosine \
max_epochs=20 \
rampdown_stop=15 \
warmup_length=1 \
rampdown_start=1 \
train_on=clothov2 \
seed=409194 \
audio_features.aggregate=TAP \
tap_hyperparams.p_dropout_weights=0.1 \
tap_hyperparams.p_dropout_proj=0.1 \
audio_features.segment_length=5 \
audio_features.hop_length=5
clotho_v2.add_phi4_captions=True
```

The expected performance is: 

| map@10 |  R@1  |  R@5  | R@10  |
|:------:|:-----:|:-----:|:-----:|
| 32.45  | 20.84 | 48.51 | 61.56 |

Training and validation took 2 hours and 55 minutes. 


## References
- [1] P. Primus, F. Schmid, and G. Widmer, “Estimated Audio-Caption Correspondences Improve Language-Based Audio Retrieval“
<a name="1"></a>
- [2] Yifei Xin, Dongchao Yang, Yuexian Zou, “Improving Text-Audio Retrieval by Text-aware Attention Pooling and Prior Matrix Revised Loss“
<a name="2"></a>
