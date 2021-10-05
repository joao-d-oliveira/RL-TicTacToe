import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
import random
import torch
import MCTS
from ConnectN import ConnectN


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        # solution
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2 * 2 * 16
        self.fc = nn.Linear(self.size, 32)

        # layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

    def forward(self, x):
        # solution
        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))

        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)
        # availability of moves
        avail = (torch.abs(x.squeeze()) != 1).type(torch.FloatTensor)
        avail = avail.reshape(1, 9)

        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail * torch.exp(a - maxa)
        prob = exp / torch.sum(exp)

        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))
        return prob.view(3, 3), value


def Policy_Player_MCTS(game, policy):
    mytree = MCTS.Node(copy(game))
    for _ in range(50):
        mytree.explore(policy)

    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)

    return mytreenext.game.last_move


def Random_Player(game):
    return random.choice(game.available_moves())


def train_policy(game_setting, policy, optimizer):
    import progressbar as pb

    # Train the policy
    episodes = 400
    outcomes = []
    losses = []

    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    for e in range(episodes):
        mytree = MCTS.Node(ConnectN(**game_setting))
        vterm = []
        logterm = []

        while mytree.outcome is None:
            for _ in range(50):
                mytree.explore(policy)

            current_player = mytree.game.player
            mytree, (v, nn_v, p, nn_p) = mytree.next()
            mytree.detach_mother()

            # solution
            # compute prob* log pi
            loglist = torch.log(nn_p) * p

            # constant term to make sure if policy result = MCTS result, loss = 0
            constant = torch.where(p > 0, p * torch.log(p), torch.tensor(0.))
            logterm.append(-torch.sum(loglist - constant))

            vterm.append(nn_v * current_player)

        # we compute the "policy_loss" for computing gradient
        outcome = mytree.outcome
        outcomes.append(outcome)

        loss = torch.sum((torch.stack(vterm) - outcome) ** 2 + torch.stack(logterm))
        optimizer.zero_grad()
        loss.backward()
        losses.append(float(loss))
        optimizer.step()

        if (e + 1) % 50 == 0:
            print("\r game: ", e + 1, ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
                  ", recent outcomes: ", outcomes[-10:], end='')
        if (e + 1) % 100 == 0: print()
        del loss

        timer.update(e + 1)

    timer.finish()