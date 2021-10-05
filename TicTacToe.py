from ConnectN import ConnectN, Play
import torch.optim as optim
from Policy import Policy, Policy_Player_MCTS, train_policy

policy = Policy()
game_setting = {'size':(3,3), 'N':3}
game = ConnectN(**game_setting)

# playing game against untrainer policy
gameplay=Play(ConnectN(**game_setting),
              player1=Policy_Player_MCTS,
              policy1=Policy(),
              player2=Policy_Player_MCTS,
              policy2=policy)

print('finished playing')
game = ConnectN(**game_setting)
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=.01, weight_decay=1.e-4)
train_policy(game_setting, policy, optimizer)

print(game_setting)
gameplay=Play(ConnectN(**game_setting),
              player1=Policy_Player_MCTS,
              policy1=Policy(),
              player2=Policy_Player_MCTS,
              policy2=policy)

print('finished playing2')

