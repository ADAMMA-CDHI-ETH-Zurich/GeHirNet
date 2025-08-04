Data_Preprocessing

1. Audiocut: segment audios into 1 sec
2. min_max: min-max on audios
3. VAD_Crossfade_based_silence_removal: voice activity detection, silence removal and mel spectrograms output

Data_Augmentation

1. Resampling: Resampling based data augmentation
2. Timewarp: Time warping based data augmentation

Training & Test

Training and test demo codes for one and two-stage architecture on epoch 30 with combination of learning rates {1e-3, 1e-4, 1e-5} and batch size {32, 64}. Two-stage test operations only focused on combining sub-models from epochs 30.

Evaluation

Evaluating the best performance of two-stage architecture via combining all best sub-models in each experiment.