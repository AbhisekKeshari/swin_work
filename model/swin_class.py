import torch
import torch.nn as nn
import timm


class swin_transformer(nn.Module):

    def __init__(self,model_name,pretrained):
        super(swin_transformer, self).__init__()
        self.model =  timm.create_model(model_name,pretrained=pretrained).cuda()
        self.avg_pool = torch.nn.AvgPool2d((98,1))

        # Special attributs
        self.input_space = None
        self.input_size = (3, 224, 224)
        self.mean = None
        self.std = None
        # Modules

        self.projection = nn.Sequential(
            nn.Linear(1536, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 1, bias=False)
        ).cuda()


    def features(self, input):  # input = (bacth, 3, 256, 256)
        x = self.model(input)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x

    def forward_one(self, input):
        x = self.features(input)
        # x = self.logits(x)

        return x

    def forward(self, input_ref, input_dis, save_output):
        x_ref = self.forward_one(input_ref)
        feat_ref = save_output.outputs[0].cuda()
        # feat_ref = torch.cat(
        #     (save_output.outputs[0],
        #      save_output.outputs[2],
        #      save_output.outputs[4],
        #      save_output.outputs[6],
        #      save_output.outputs[8],
        #      save_output.outputs[10]),
        #     dim=1
        # )  # feat_ref: n_batch x (320*6) x 29 x 29
        # clear list (for saving feature map of d_img)
        save_output.outputs.clear()
        x_dis = self.forward_one(input_dis)
        feat_dis = save_output.outputs[0].cuda()
        # feat_dis = torch.cat(
        #     (save_output.outputs[0],
        #      save_output.outputs[2],
        #      save_output.outputs[4],
        #      save_output.outputs[6],
        #      save_output.outputs[8],
        #      save_output.outputs[10]),
        #     dim=1
        # )  # feat_ref: n_batch x (320*6) x 29 x 29
        # clear list (for saving feature map of r_img in next iteration)
        save_output.outputs.clear()

        input = torch.cat((feat_ref,feat_dis),dim=1).cuda()

        # input = input.permute(0, 2, 1)[:, :, -1]
        input = self.avg_pool(input)[:,0,:]

        pred = self.projection(input)
        # x = self.logits(x)
        # x = torch.abs(x_ref - x_dis)
        return pred


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []