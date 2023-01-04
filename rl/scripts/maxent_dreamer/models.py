from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    if isinstance(y, tuple):
        q1, q2 = y
        q1_size = q1.size()
        q2_size = q2.size()
        output1 = q1.view(x_sizes[0][0], x_sizes[0][1], *q1_size[1:])
        output2 = q2.view(x_sizes[0][0], x_sizes[0][1], *q2_size[1:])
        return  output1, output2

    else:
        y_size = y.size()
        output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
        return output


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        # state: s_t
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior,
                        self.fc_embed_belief_posterior, self.fc_state_posterior]

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor,
                observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None) -> List[
        torch.Tensor]:
        '''
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs =\
            [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, \
            [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T

        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[
                t]  # Select appropriate previous state
            _state = _state if nonterminals is None else _state * nonterminals[
                t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
                  torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class SymbolicObservationModel(jit.ScriptModule):
    # decoder for symbolic input
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(jit.ScriptModule):
    # decoder for image input.
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    # this part is the decoder model.
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class ValueModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x), inplace=True)
        hidden = self.act_fn(self.fc2(hidden),inplace=True)
        hidden = self.act_fn(self.fc3(hidden),inplace=True)
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ActorModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                 activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
        '''
        the belief size indicates the gru's hidden size, the hidden size indicates the latent size.
        '''
        '''
        belief : h_t
        state: l_t
        s_t = cat([h_t, l_t])
        actor model: a_t ~ q(a_t|s_t)
        '''
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, det=False):
        # can't fully understand this parts.
        action_mean, action_std = self.forward(belief, state)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)
        if det:
            return dist.mode()
        else:
            return dist.rsample()

class ActorSAC(jit.ScriptModule):
    def __init__(self,belief_size, state_size, hidden_size, action_size, if_use_dn=False):
        super().__init__()
        state_dim = belief_size + state_size
        if if_use_dn:
            nn_middle = DenseNet(hidden_size // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size), nn.ReLU(), )
            inp_dim = hidden_size
            out_dim = hidden_size

        self.net_state = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                       nn_middle, )
        self.net_a_avg = nn.Sequential(nn.Linear(out_dim, hidden_size), nn.Hardswish(),
                                       nn.Linear(hidden_size, action_size))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(out_dim, hidden_size), nn.Hardswish(),
                                       nn.Linear(hidden_size, action_size))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    @jit.script_method
    def forward(self,  belief, state):
        x = torch.cat([belief, state], dim=1)
        tmp = self.net_state(x)
        return self.net_a_avg(tmp).tanh()  # action


    def get_action(self, belief, state, det=None):
        x = torch.cat([belief, state], dim=1)
        t_tmp = self.net_state(x)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize


    def get_action_logprob(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        t_tmp = self.net_state(x)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_avg, a_std).log_prob(a_noise)
        # log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        # same as below:
        # a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_tan, log_prob.sum(1, keepdim=True).squeeze()


class CriticTwin(jit.ScriptModule):  # shared parameter
    def __init__(self, belief_size, state_size,action_size, hidden_size, if_use_dn=False):
        super().__init__()
        state_dim = belief_size + state_size +action_size
        if if_use_dn:
            nn_middle = DenseNet(hidden_size // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), )

            inp_dim = hidden_size
            out_dim = hidden_size
        self.net_sa = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                    nn_middle, )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(out_dim, hidden_size), nn.Hardswish(),
                                    nn.Linear(hidden_size, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(out_dim, hidden_size), nn.Hardswish(),
                                    nn.Linear(hidden_size, 1))  # q2 value
        self.modules = [self.net_sa, self.net_q1, self.net_q2]
    @jit.script_method
    def forward(self, belief, state, action):
        tmp = self.net_sa(torch.cat((belief, state, action), dim=1))
        return self.net_q1(tmp).squeeze(dim=1),self.net_q2(tmp).squeeze(dim=1)  # one Q value


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(jit.ScriptModule):
    __constants__ = ['embedding_size']
    def __init__(self, embedding_size, activation_function='relu'):
        # an simple auto encoder.
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        '''
        the nn.Identity() 's foward:
        >>> forward(inputs):
                return inputs
        this func is a coding trick for keeping layer size.  
        '''
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    # encoder  the obs into latent vector, the latent size is : embedding_size.
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()

class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)
