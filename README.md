# About

Official PyTorch implementation of the ET DeepFool algorithm from the paper, "Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm".

## Setup

We encourage creating a virtual enviornment to run our code.

### Requirements

```
torch == 2.0.0+cu117  

torchaudio == 2.0.1+cu117  

torchmetrics == 0.11.4  

torchvision == 0.15.1+cu117  

matplotlib == 3.7.1
```

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

* Published on Nature Scientific Reports. [Link](https://www.nature.com/articles/s41598-025-87405-w).

## References

If you have used this code, please cite the following paper:

[1] S. M. F. R. Labib, J. J. Mondal, M. A. Manab, X. Xiao, and S. Newaz, “Tailoring adversarial attacks on deep neural networks for targeted class manipulation using DeepFool algorithm,” Scientific Reports, vol. 15, no. 1, p. 10790, 2025, doi: 10.1038/s41598-025-87405-w.

```bibtex
@article{Labib_2025,
    title = {Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm},
    year = {2025},
    ISSN = {2045-2322},
    url = {https://doi.org/10.1038/s41598-025-87405-w},
    DOI = {10.1038/s41598-025-87405-w},
    number = {1},
    volume = {15},
    journal = {Scientific Reports},
    publisher = {Springer Science and Business Media LLC},
    author = {Labib, S M Fazle Rabby and Mondal, Joyanta Jyoti and Manab, Meem Arafat and Xiao, Xi and Newaz, Sarfaraz},
}
```
