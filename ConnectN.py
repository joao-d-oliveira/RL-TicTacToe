import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from copy import copy


# output the index of when v has a continuous string of i
# get_runs([0,0,1,1,1,0,0],1) gives [2],[5],[3]
def get_runs(v, i):
     bounded = np.hstack(([0], (v==i).astype(int), [0]))
     difs = np.diff(bounded)
     starts, = np.where(difs > 0)
     ends, = np.where(difs < 0)
     return starts, ends, ends-starts


# see if vector contains N of certain number in a row
def in_a_row(v, N, i):
     if len(v) < N:
          return False
     else:
          _, _, total = get_runs(v,i)
          return np.any(total >= N) 
     

def get_lines(matrix, loc):

     i,j=loc
     flat = matrix.reshape(-1,*matrix.shape[2:])
    
     w = matrix.shape[0]
     h = matrix.shape[1]
     def flat_pos(pos):
          return pos[0]*h+pos[1]

     pos = flat_pos((i,j))

     # index for flipping matrix across different axis
     ic = w-1-i
     jc = h-1-j

     # top left
     tl = (i-j,0) if i>j else (0, j-i)
     tl = flat_pos(tl)

     # bottom left
     bl = (w-1-(ic-j),0) if ic>j else (w-1, j-ic)
     bl = flat_pos(bl)

     # top right
     tr = (i-jc,h-1) if i>jc else (0, h-1-(jc-i))
     tr = flat_pos(tr)

     # bottom right
     br = (w-1-(ic-jc),h-1) if ic>jc else (w-1, h-1-(jc-ic))
     br = flat_pos(br)

     hor = matrix[:,j]
     ver = matrix[i,:]
     diag_right = np.concatenate([flat[tl:pos:h+1],flat[pos:br+1:h+1]])
     diag_left = np.concatenate([flat[tr:pos:h-1],flat[pos:bl+1:h-1]])

     return hor, ver, diag_right, diag_left


class ConnectN:

     def __init__(self, size, N, pie_rule=False):
          self.size = size
          self.w, self.h = size
          self.N = N

          # make sure game is well defined
          if self.w<0 or self.h<0 or self.N<2 or \
             (self.N > self.w and self.N > self.h):
               raise ValueError('Game cannot initialize with a {0:d}x{1:d} grid, and winning condition {2:d} in a row'.format(self.w, self.h, self.N))

          
          self.score = None
          self.state=np.zeros(size, dtype=np.float)
          self.player=1
          self.last_move=None
          self.n_moves=0
          self.pie_rule=pie_rule
          self.switched_side=False

     # fast deepcopy
     def __copy__(self):
          cls = self.__class__
          new_game = cls.__new__(cls)
          new_game.__dict__.update(self.__dict__)

          new_game.N = self.N
          new_game.pie_rule = self.pie_rule
          new_game.state = self.state.copy()
          new_game.switched_side = self.switched_side
          new_game.n_moves = self.n_moves
          new_game.last_move = self.last_move
          new_game.player = self.player
          new_game.score = self.score
          return new_game
    
     # check victory condition
     # fast version
     def get_score(self):

          # game cannot end beca
          if self.n_moves<2*self.N-1:
               return None

          i,j = self.last_move
          hor, ver, diag_right, diag_left = get_lines(self.state, (i,j))

          # loop over each possibility
          for line in [ver, hor, diag_right, diag_left]:
               if in_a_row(line, self.N, self.player):
                    return self.player
                    
          # no more moves
          if np.all(self.state!=0):
               return 0

          return None

     # for rendering
     # output a list of location for the winning line
     def get_winning_loc(self):
        
          if self.n_moves<2*self.N-1:
               return []

          
          loc = self.last_move
          hor, ver, diag_right, diag_left = get_lines(self.state, loc)
          ind = np.indices(self.state.shape)
          ind = np.moveaxis(ind, 0, -1)
          hor_ind, ver_ind, diag_right_ind, diag_left_ind = get_lines(ind, loc)
          # loop over each possibility
        
          pieces = [hor, ver, diag_right, diag_left]
          indices = [hor_ind, ver_ind, diag_right_ind, diag_left_ind]
        
          #winning_loc = np.full(self.state.shape, False, dtype=bool)
        
          for line, index in zip(pieces, indices):
               starts, ends, runs = get_runs(line, self.player)

               # get the start and end location
               winning = (runs >= self.N)
               print(winning)
               if not np.any(winning):
                    continue
            
               starts_ind = starts[winning][0]
               ends_ind = ends[winning][0]
               indices = index[starts_ind:ends_ind]
               #winning_loc[indices[:,0], indices[:,1]] = True
               return indices
            
          return []
    
    
     def move(self, loc):
          i,j=loc
          success = False
          if self.w>i>=0 and self.h>j>=0:
               if self.state[i,j]==0:

                    # make a move
                    self.state[i,j]=self.player

                    # if pie rule is enabled
                    if self.pie_rule:
                         if self.n_moves==1:
                              self.state[tuple(self.last_move)]=-self.player
                              self.switched_side=False
                    
                         elif self.n_moves==0:
                              # pie rule, make first move 0.5
                              # this is to let the neural net know
                              self.state[i,j]=self.player/2.0
                              self.switched_side=False
                         
                    success = True

               # switching side
               elif self.pie_rule and self.state[i,j] == -self.player/2.0:

                    # make a move
                    self.state[i,j]=self.player
                    self.switched_side=True

                    success = True

                         
               

          if success:
               self.n_moves += 1
               self.last_move = tuple((i,j))
               self.score = self.get_score()

               # if game is not over, switch player
               if self.score is None:
                    self.player *= -1
               
               return True

          return False
    
    
     def available_moves(self):
          indices = np.moveaxis(np.indices(self.state.shape), 0, -1)
          return indices[np.abs(self.state) != 1]

     def available_mask(self):
          return (np.abs(self.state) != 1).astype(np.uint8)


class Play:

     def __init__(self, game, player1=None, player2=None, policy1=None, policy2=None, name='game'):
          self.original_game = game
          self.game = copy(game)
          self.player1 = player1
          self.policy1 = policy1
          self.player2 = player2
          self.policy2 = policy2
          self.player = self.game.player
          self.end = False
          self.play()

     def reset(self):
          self.game = copy(self.original_game)
          self.click_cid = None
          self.end = False

     def play(self, name='Game'):

          self.reset()

          if self.game.w * self.game.h < 25:
               figsize = (self.game.w / 1.6, self.game.h / 1.6)

          else:
               figsize = (self.game.w / 2.1, self.game.h / 2.1)

          self.fig = plt.figure(name, figsize=figsize)
          if self.game.w * self.game.h < 25:
               self.fig.subplots_adjust(.2, .2, 1, 1)
          else:
               self.fig.subplots_adjust(.1, .1, 1, 1)

          self.fig.show()
          w, h = self.game.size
          self.ax = self.fig.gca()
          self.ax.grid()
          # remove hovering coordinate tooltips
          self.ax.format_coord = lambda x, y: ''
          self.ax.set_xlim([-.5, w - .5])
          self.ax.set_ylim([-.5, h - .5])
          self.ax.set_xticks(np.arange(0, w, 1))
          self.ax.set_yticks(np.arange(0, h, 1))
          self.ax.set_aspect('equal')

          for loc in ['top', 'right', 'bottom', 'left']:
               self.ax.spines[loc].set_visible(False)

          # fully AI game
          if self.player1 is not None and self.player2 is not None:
               self.anim = FuncAnimation(self.fig, self.draw_move, frames=self.move_generator, interval=500,
                                         repeat=False)
               return

          # at least one human
          if self.player1 is not None:
               # first move from AI first
               succeed = False
               while not succeed:
                    loc = self.player1(self.game, self.policy1) if self.policy1 is not None else self.player1(self.game)
                    succeed = self.game.move(loc)

               self.draw_move(loc)

          self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.click)

     def move_generator(self):
          score = None
          # game not concluded yet
          while score is None:
               self.player = self.game.player
               if self.game.player == 1:
                    loc = self.player1(self.game, self.policy1) if self.policy1 is not None else self.player1(self.game)
               else:
                    loc = self.player2(self.game, self.policy2) if self.policy2 is not None else self.player2(self.game)

               success = self.game.move(loc)

               # see if game is done
               if success:
                    score = self.game.score
                    yield loc

     def draw_move(self, move=None):
          if self.end:
               return

          i, j = self.game.last_move if move is None else move
          c = 'salmon' if self.player == 1 else 'lightskyblue'
          self.ax.scatter(i, j, s=500, marker='o', zorder=3, c=c)
          score = self.game.score
          self.draw_winner(score)
          self.fig.canvas.draw()

     def draw_winner(self, score):
          if score is None:
               return

          if score == -1 or score == 1:
               locs = self.game.get_winning_loc()
               c = 'darkred' if score == 1 else 'darkblue'
               self.ax.scatter(locs[:, 0], locs[:, 1], s=300, marker='*', c=c, zorder=4)

          # try to disconnect if game is over
          if hasattr(self, 'click_cid'):
               self.fig.canvas.mpl_disconnect(self.click_cid)

          self.end = True

     def click(self, event):

          loc = (int(round(event.xdata)), int(round(event.ydata)))
          self.player = self.game.player
          succeed = self.game.move(loc)

          if succeed:
               self.draw_move()

          else:
               return

          if self.player1 is not None or self.player2 is not None:

               succeed = False
               self.player = self.game.player
               while not succeed:
                    if self.game.player == 1:
                         loc = self.player1(self.game, self.policy1) if self.policy1 is not None else self.player1(
                              self.game)
                    else:
                         loc = self.player2(self.game, self.policy2) if self.policy2 is not None else self.player2(
                              self.game)

                    succeed = self.game.move(loc)

               self.draw_move()
