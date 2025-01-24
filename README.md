# About

Official PyTorch implementation of the ET DeepFool algorithm from the paper, "Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm".

## Setup

We encourage creating a virtual enviornment to run our code.

### Requirements

torch == 2.0.0+cu117  

torchaudio == 2.0.1+cu117  

torchmetrics == 0.11.4  

torchvision == 0.15.1+cu117  

matplotlib == 3.7.1

### Issues you might run into

You might run into a problem when importing the libraries.

The following error might show up: **"cannot import name 'zero_gradients' from 'torch.autograd.gradcheck"**

There will be a path at the end of the error, which should lead you to `gradcheck.py` file

Copy and paste the following code the `gradcheck.py` file

``` python
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
```

Save the `gradcheck.py` file and you should be good to go.

## Paper

* [Preprint](https://arxiv.org/abs/2310.13019)
* This work has been accepted for publication in _Nature Scientific Reports_. To be added here [TBA](#).

## References

If you have used this code, please cite the following paper:

[1] Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm. S. M. Fazle Rabby Labib, Joyanta Jyoti Mondal, Meem Arafat Manab, Xi Xiao, and Sarfaraz Newaz.

```bibtex
@misc{labib2024tailoringadversarialattacksdeep,
      title={Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm}, 
      author={S. M. Fazle Rabby Labib and Joyanta Jyoti Mondal and Meem Arafat Manab and Xi Xiao and Sarfaraz Newaz},
      year={2024},
      eprint={2310.13019},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.13019}, 
}
```
