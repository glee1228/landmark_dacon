from collections import OrderedDict
from torchvision import models
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# landmark => 1049 class

class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            output = self.__class__.__name__ + "\n"
            output += self._summary(self, input_size, batch_size, device)
            return output
        except:
            return self.__repr__()

    @staticmethod
    def _summary(model, input_size, batch_size=-1, device="cuda"):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                params = 0

                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad

                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))

                summary[m_key]["nb_params"] = params

            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)
                    and not (module == model)):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        # print(type(x[0]))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()
        output = "---------------------------------------------------------------------------------------\n"
        output += "{:^30}{:^30}{:^15}{:^5}\n".format("Layer (type)", "Output Shape", "Param #", "Grad")
        output += "=======================================================================================\n"
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]

            output += "{:^30}{:^30}{:^15}{:^5}\n".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
                "True" if summary[layer].get("trainable") else "False"
            )

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        output += "=======================================================================================\n"
        output += "Total params: {0:,}\n".format(total_params)
        output += "Trainable params: {0:,}\n".format(trainable_params)
        output += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
        output += "---------------------------------------------------------------------------------------\n"
        output += "Input size (MB): %0.2f\n" % total_input_size
        output += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
        output += "Params size (MB): %0.2f\n" % total_params_size
        output += "Estimated Total Size (MB): %0.2f\n" % total_size
        output += "---------------------------------------------------------------------------------------"

        return output


class Resnet50(BaseModel):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1049)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.fc = nn.Linear(64, 1049)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.AdaptiveAvgPool2d(1)(x).squeeze()
        x = self.fc(x)
        return x


class Efficientnet(BaseModel):
    def __init__(self,depth):
        super(Efficientnet, self).__init__()
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{depth}', num_classes=1049)

    def forward(self, x):
        x = self.base(x)
        return x


if __name__ == '__main__':
    # m = Efficientnet(4)
    m=Resnet50()
    grad = False
    for n, p in m.named_parameters():
        p.requires_grad = grad = grad or n.startswith('base._blocks.22')
        print(n, p.requires_grad)

    print(m.summary((3, 224, 224), device='cpu'))
