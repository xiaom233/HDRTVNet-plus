import torch.nn as nn


def color_block(in_filters, out_filters, normalization=False):
    conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv, pooling, act]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


# Release Version
class Color_Condition(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


# Function ablation of Condition
# -------------------------------------------------------------------------------
class Color_Condition_woDropout(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_woDropout, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            # nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_Condition_woIN(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_woIN, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=False),
            *color_block(16, 32, normalization=False),
            *color_block(32, 64, normalization=False),
            *color_block(64, 128, normalization=False),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)
# -------------------------------------------------------------------------------



# Depth Comparison of Condition
# -------------------------------------------------------------------------------
# Release Version
class Color_Condition_3layer(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_3layer, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 32),
            nn.Dropout(p=0.5),
            nn.Conv2d(32, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_Condition_4layer(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_4layer, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 64),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_Condition_6layer(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_6layer, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 256, normalization=True),
            *color_block(256, 256),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)

# -------------------------------------------------------------------------------


class ConditionNet(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet, self).__init__()
        self.classifier = classifier
        if self.classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out, x


# 3 layers base model
class BaseModel(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.HRconv(out)
        out = self.act(out)
        out = self.conv_last(out)

        return out


# Depth Comparison of Base Model
# -------------------------------------------------------------------------------
class BaseModel2layer(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel2layer, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.conv_last(out)

        return out


class BaseModel4layer(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel4layer, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.HRconv(out)
        out = self.act(out)
        out = self.conv_last1(out)
        out = self.act(out)
        out = self.conv_last2(out)

        return out


class BaseModel5layer(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel5layer, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last3 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.HRconv(out)
        out = self.act(out)
        out = self.conv_last1(out)
        out = self.act(out)
        out = self.conv_last2(out)
        out = self.act(out)
        out = self.conv_last3(out)
        return out


# Depth Comparison of Condition
# -------------------------------------------------------------------------------
class ConditionNet2Layer(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet2Layer, self).__init__()
        if classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        else:
            raise

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        print(self.classifier)
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, 3, 1, 1) + shift_HR.view(-1, 3, 1, 1) + out

        return out


class ConditionNet4Layer(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet4Layer, self).__init__()
        if classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        else:
            raise
        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last1 = nn.Linear(cond_c, nf)
        self.cond_scale_last2 = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last1 = nn.Linear(cond_c, nf)
        self.cond_shift_last2 = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last1 = self.cond_scale_last1(fea)
        shift_last1 = self.cond_shift_last1(fea)

        scale_last2 = self.cond_scale_last2(fea)
        shift_last2 = self.cond_shift_last2(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last1(out)
        out = out * scale_last1.view(-1, self.GFM_nf, 1, 1) + shift_last1.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last2(out)
        out = out * scale_last2.view(-1, 3, 1, 1) + shift_last2.view(-1, 3, 1, 1) + out

        return out


class ConditionNet5Layer(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet5Layer, self).__init__()
        if classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        else:
            raise
        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last1 = nn.Linear(cond_c, nf)
        self.cond_scale_last2 = nn.Linear(cond_c, nf)
        self.cond_scale_last3 = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last1 = nn.Linear(cond_c, nf)
        self.cond_shift_last2 = nn.Linear(cond_c, nf)
        self.cond_shift_last3 = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last3 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last1 = self.cond_scale_last1(fea)
        shift_last1 = self.cond_shift_last1(fea)

        scale_last2 = self.cond_scale_last2(fea)
        shift_last2 = self.cond_shift_last2(fea)

        scale_last3 = self.cond_scale_last3(fea)
        shift_last3 = self.cond_shift_last3(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last1(out)
        out = out * scale_last1.view(-1, self.GFM_nf, 1, 1) + shift_last1.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last2(out)
        out = out * scale_last2.view(-1, self.GFM_nf, 1, 1) + shift_last2.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last3(out)
        out = out * scale_last3.view(-1, 3, 1, 1) + shift_last3.view(-1, 3, 1, 1) + out

        return out
# -------------------------------------------------------------------------------



