import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach() #change requires_grad = flase

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

class Action_encoder(nn.Module):
    def __init__(self, action_shape, feature_num=10, num_layers=2, output_logits=False):
        super().__init__()

        self.action_shape = action_shape

        self.feature_num = feature_num
        #self.num_layers = num_layers
        self.output_logits = output_logits

        self.net = nn.Sequential(
            nn.Linear(self.action_shape,512), nn.ReLU(),
            nn.Linear(512,self.feature_num)
        )

        self.ln = nn.LayerNorm(self.feature_num)
        self.output_logits = output_logits
        #self.apply(weight_init)

    def forward(self,actions,detach=False):

        output = self.net(actions)

        if detach:
            output = output.detach()

        output_norm = self.ln(output)

        #normalizes input across the features
        if self.output_logits:
            output = output_norm
            #mean = torch.mean(output, dim=1)
            #std = torch.std(output, dim=1)
        else:
            output = torch.tanh(output_norm)

        return output

class Transition_model(nn.Module):

    def __init__(self,latent_action_shape, latent_feature_shape, output_feature_num=50, num_layers=2, output_logits=False):
        super().__init__()
        self.latent_action_shape = latent_action_shape
        self.latent_feature_shape = latent_feature_shape
        self.output_feature_num = output_feature_num
        self.num_layers = num_layers
        self.output_logits = output_logits

        self.net = nn.Sequential(
            nn.Linear(self.latent_action_shape + self.latent_feature_shape, 1024), nn.ReLU(),
            nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024,self.output_feature_num),
        )

        self.ln = nn.LayerNorm(self.output_feature_num)
        #self.apply(weight_init)

    def forward(self, latent_obs, latent_actions, detach = False):
        obs_actions = torch.cat([latent_obs, latent_actions], dim=1)

        output = self.net(obs_actions)

        if detach:
            output = output.detach()

        output_norm = self.ln(output)

        if self.output_logits:
            output = output_norm
        else:
            output = torch.tanh(output_norm)

        return output


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
