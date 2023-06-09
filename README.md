# Repository for MS Thesis

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Datasets

The thesis uses three datasets: [L2Arctic](https://psi.engr.tamu.edu/l2-arctic-corpus/), [TORGO](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html), and [UA-Speech](http://www.isle.illinois.edu/sst/data/UASpeech/).

## Getting started

To download and use Conda:

1. `wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh`

2. `sh Anaconda3-2022.10-Linux-x86_64.sh`

> Note: The scripts to train and submit to HTCondor use `source ~/.bash_profile`. If `sh Anaconda3-2022.10-Linux-x86_64.sh` creates or modifies a `.bashrc` file, then move it's created contents to `source ~/.bash_profile`.
>
> A sample `.bash_profile` will look like:

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home2/hsteinm/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home2/hsteinm/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home2/hsteinm/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home2/hsteinm/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

3. `rm Anaconda3-2022.10-Linux-x86_64.sh`

4. `source ~/.bash_profile`

5. `conda env create -f environment-linux.yml`