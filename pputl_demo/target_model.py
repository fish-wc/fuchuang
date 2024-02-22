import math
import torch.nn.init as init
from pputl_demo.source_model import *
from functools import partial
from pputl_demo.component import *

channel = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    def __init__(self, image_size=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.linear_1 = nn.Linear(128, 4 * 4 * 256)
        self.linear_2 = nn.Linear(10, 4 * 4 * 256)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Self_Attn(128, 'relu'),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Self_Attn(64, 'relu'),
            nn.ConvTranspose2d(64, channel, 4, 2, 1),
            nn.Tanh(),
        )
        res_arch_init(self)

    def forward(self, z, y):

        x = self.linear_1(z)
        y = self.linear_2(y)
        x = torch.cat([x, y], 1)
        x = x.view(-1, 512, 4, 4)
        x = self.layers(x)

        return x


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.1),
            # Self_Attn(256, 'relu'),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.1),
            # Self_Attn(512, 'relu'),
        )
        self.linear=nn.Linear(4*4*512,1)
        res_arch_init(self)
        
    def forward(self, x):

        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

        
def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)


class Generator64(Generator):
    def __init__(self):
        super().__init__()


class Discriminator64(Discriminator):
    def __init__(self):
        super().__init__()


class Wasserstein(nn.Module):

    def cacl_gradient_penalty(self, net_D, real, fake):
        t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
        t = t.expand(real.size())

        interpolates = t * real + (1 - t) * fake
        interpolates.requires_grad_(True)
        disc_interpolates = net_D(interpolates)
        grad = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, retain_graph=True)[0]

        grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
        loss_gp = torch.mean((grad_norm - 1) ** 2)
        return loss_gp

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class l2_norm(nn.Module):
    def forward(self, theta1, theta2):
        sum_all = 0
        for key in theta1.keys():
            sum_all += torch.sum(torch.square(theta1[key] - theta2[key]))

        return torch.sqrt(sum_all)


class self_entropy(nn.Module):
    def forward(self, x):
        n = x.size(0)
        loss = torch.sum(-x * torch.log(x + 10 ** -10)) / n
        return loss


class feature_dis(nn.Module):
    def forward(self, x, y):
        n = x.size(0)
        loss = 0
        for i in range(n):
            loss += torch.norm(x[i] - y[i])
        loss = loss / n
        return loss


class MMD_loss(nn.Module):
    def pairwise_distance(self, x, y):
        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)

        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)

        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))

        return cost

    def forward(self, source_features, target_features):
        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
            self.gaussian_kernel_matrix, sigmas=torch.FloatTensor(sigmas).to(device)
        )
        loss_value = self.maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
        loss_value = loss_value

        return loss_value


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def _disable_tracking_bn_stats(self, extractor, classifier):

        def switch_attr(m):
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats ^= True

        extractor.apply(switch_attr)
        classifier.apply(switch_attr)
        yield
        extractor.apply(switch_attr)
        classifier.apply(switch_attr)

    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def forward(self, extractor, classifier, x):
        with torch.no_grad():
            pred = F.softmax(classifier(extractor(x)), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = self._l2_normalize(d)

        with self._disable_tracking_bn_stats(extractor, classifier):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = classifier(extractor(x + self.xi * d))
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = self._l2_normalize(d.grad)
                extractor.zero_grad()
                classifier.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = classifier(extractor(x + r_adv))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
