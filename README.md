> @author: Yuanhang Tang

> 2021/08/30
- Upload my codes
- what to do next:
    - preprocess the text to be a `torch.Tensor`
    - modify the dataset design to make it compatible with torch `DataLoader`
    - implement a transformer, or at least clone one :)
    - Maybe I need to try an alternative way to store my dataset
    
- `[cls]`, `[sep]`, `[pad]` == 101, 102, 0

> 2021/08/31
- feeling confused of about how I should modify my implementation ... Hard work to make it
  both readable and efficient
  
- I finally deicided to pad the docs to make it a legal `torch.Tensor`

> 2021/09/01
- modify the argument settings of the models to facilitate data loading
- rewrite the loading mechanism (`DataLoader`)
- modify the implementation of `Validator`
- test `Model`
- EDA: median of doc length is 37, the 0.9 quantile is 94. Maybe we can set it to be 64?
- EDA: most of the sentences have less than 100 characters; 128 may works pretty well

> 2021/09/02
- define the labeled loss
  - TSA: training signal annealing
  - the implementation of `ClassificationLoss` is inspired by `lr_schedulers` in `PyTorch`
  
- together with the unlabeled loss:
  - prediction sharpening: I will leave this for tomorrow because I am NOT clear 
    about what this is
    
> 2021/09/03
- define the consistency loss
- Back-translation by Baidu api finish its first stage
- Oooooooooops! Something's wrong with my data
  - storing corpus with `csv` file may lead to errors ...
  - modify data storing methods
- modify `config` settings
- Samples that cause in the back-translation still cause error ... Maybe they are too long ???? 

> 2021/09/04
- construct a dataset using those `succFiles`
- some files are incomplete ... DROP them
- some seem to be wrong ??
  - repeating words ...
  - '\n' cut
  - some translation is really funny ... hhhhhhhhh ...
  
- IT IS SO HARD FOR A SINGLE DEVELOPER TO REPEAT THE WORKS OF THOSE BIG COMPANIES!!!!
  - Computing resource: limited
  - Workload: HUGE
  - Coding skills: poor ...
  - tools support: build/find by yourself
  - Money: damn it ...
  
> 2021/09/05
- do a lot of date preprocessing
- modify the implementation of the `dataset`
- tips:
  - preprocess your dataset to an appropriate form, and store them in an approriate data format
    - corpus in `json`
    - numerical features in `csv`
  - make sub-folders in `data`: `debug`, `train`, `test`
- I meet with some choices:
  - I may preprocess my data when reading them into my `Dataset`, but different datasets may share
    the same preprocessing tools