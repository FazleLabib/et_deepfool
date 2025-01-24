import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

def deepfool_et(image, net, target_class, overshoot=0.02, min_confidence = 95):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param target_class: target class that the image should be misclassified as
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param min_confidence: minimum amount of confidence for ET Deepfool
       :return: minimal perturbation that fools the classifier, number of iterations that it required, target label, perturbed image, and confidence scores
    """

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label
    confidence = 0

    while k_i != target_class or confidence < min_confidence:

        fs[0, label].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        zero_gradients(x)

        fs[0, target_class].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()

        w_k = cur_grad - grad_orig
        f_k = (fs[0, target_class] - fs[0, label]).data.cpu().numpy()

        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

        w = (pert_k + 1e-4) * w_k / np.linalg.norm(w_k)

        r_i = (1 + overshoot) * w
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        
        confidence = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item() * 100

        loop_i += 1
    
    r_tot = (1+overshoot)*r_tot

    confidence_target = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item()
    confidence_orig = torch.nn.functional.softmax(fs, dim=1)[0, label].item()

    orig_image = image.cpu().numpy().flatten()
    perturbed_image = pert_image.cpu().numpy().flatten()
    l2_dist = np.linalg.norm(perturbed_image - orig_image)
    max_l2_dist = np.sqrt(np.prod(input_shape))
    change = (l2_dist / max_l2_dist)

    ssim = structural_similarity_index_measure(image.to('cuda').unsqueeze(0), pert_image)
    ssim_value = round(ssim.item(), 4)

    return r_tot, loop_i, label, k_i, pert_image, confidence_target, confidence_orig, change, ssim_value