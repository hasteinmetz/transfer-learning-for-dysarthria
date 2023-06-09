# Source code for the thesis project

## Experiment matrix

| Experiment                   | Out-of-the-box | Finetune--Dysarthria | Finetune--All | Multitask--Same Layer | Multitask--Separate Layer |
|------------------------------|----------------|----------------------|---------------|-----------------------|---------------------------|
| Speaker-dependent            | -              | -                    | -             | -                     | -                         |
| UASpeech $\rightarrow$ TORGO | -              | -                    | -             | -                     | -                         |
| Zero-shot                    | -              | -                    | -             | -                     | X                         |

## TO DO

*ADD WARMUP PARAMETERS* -- This should help with the issue of overfitting deep networks

### SMALL STUFF

- Automatically infer whether to restart a training sequence or not 
    - Base this on whether the previous checkpoint is close or not 
- Create list of valid arguments + explanations in the config folder readme
- Delete the unused audio files as well to save space (compressed instead)
- Print metrics at the end of each epoch /
- Use torch script jit for speed
- Automatically determine batch size depending on CUDA memory available
- Manually override config file so that (for instance) you can load a model

### PRIORITIES

- Figure out a way to automatically detect anomalies in speech files
- Fix the dataset ratios recorded in the txt file
- Start working on an intermediate fine-tuning script
  or a metalearning script
- **Come up with a audio complete file creation script**
- **Visualize the data with histograms to determine best percentile!!**

### Resolve finetuning issue

- Last 1-3 layers of wav2vec2.0 are not well-initialized (Pased et al.)... Maybe re-initialize those randomly before finetuning


### DONE

- Add train and eval args to config file
- Add thing to determine whether GPU arg provided to HTCondor submit file
- Fixed dataset loading
- Run [noise_reduce](https://github.com/timsainb/noisereduce) algorithm on the TORGO dataset
- Figure out a way to use a config for number of linear layer heads
- Add a regularization term to properly weight the datasets (seems like L2Arctic is weighted too high)
    > Make the Multitask class adaptable to mulitple tasks (i.e. automatically create the correct number of layers)
    > Make the load_model function sensitive to number of tasks and instantiated a model that way

For environment:

 && pip install certifi==2022.12.7 datasets==2.8.0 dill==0.3.4 noisereduce==2.0.1 pandas==1.5.2 pyarrow==10.0.1 xxhash==3.2.0 && conda install matplotlib
