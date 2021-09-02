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
  - the implementation of `TSALoss` is inspired by `lr_schedulers` in `PyTorch`
- together with the unlabeled loss:
  - prediction sharpening: I will leave this for tomorrow because I am NOT clear about what this is