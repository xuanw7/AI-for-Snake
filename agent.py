import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):

        self.points = 0
        self.s = None
        self.a = None
    
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''

        # food_dir_x, food_dir_y, 
        # adjoining_wall_x, adjoining_wall_y, 
        # adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right
        sp= self.generate_state(environment)

            

        # update Q_value
        if self.s != None:
            max_Q_sp_a = float('-inf')

            for i in range(4):
                Q_sp_a = self.Q[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]][sp[5]][sp[6]][sp[7]][i]
                if (Q_sp_a > max_Q_sp_a):
                    max_Q_sp_a = Q_sp_a

            s = self.s
            Q_s_a = self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][self.a]
            self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][self.a] += 1
            alpha = self.C / (self.C + self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][self.a])

            r = -0.1
            if (self.points < points):
                r = 1
            if (dead):
                r= -1
            self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][self.a] = Q_s_a + alpha * (r + self.gamma * max_Q_sp_a - Q_s_a)

        if dead :
            self.reset()
            return 0
        



        self.s = sp
        self.points = points


        #choose action
        a = 0
        max_Q_s_a = float('-inf')
        flag = True
        for i in range(4):
            N_s_a = self.N[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]][sp[5]][sp[6]][sp[7]][i]
            if (N_s_a < self.Ne):
                a = i
                flag = False
            elif (flag):
                Q_s_a = self.Q[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]][sp[5]][sp[6]][sp[7]][i]
                if (Q_s_a >= max_Q_s_a):
                    max_Q_s_a = Q_s_a
                    a = i
        self.a = a

        return a


        

    def generate_state(self, environment):


        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]

        food_dir_x = 0
        if (food_x < snake_head_x):
            food_dir_x = 1
        elif (food_x > snake_head_x):
            food_dir_x = 2
        else:
            food_dir_x = 0

        food_dir_y = 0
        if (food_y < snake_head_y):
            food_dir_y = 1
        elif (food_y > snake_head_y):
            food_dir_y = 2
        else:
            food_dir_y = 0

        height = utils.DISPLAY_HEIGHT
        width = utils.DISPLAY_WIDTH 


        adjoining_wall_x = 0

        if (snake_head_x - 1 == 0):
            adjoining_wall_x = 1
        elif (snake_head_x + 1 == width - 1):
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0
        
        
        
        adjoining_wall_y = 0

        if (snake_head_y - 1 == 0):
            adjoining_wall_y = 1
        elif (snake_head_y + 1 == height - 1):
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0


        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        x = snake_head_x
        y = snake_head_y - 1
        for i in range(len(snake_body)):
            if snake_body[i] == (x,y):
                adjoining_body_top = 1
                break

        x = snake_head_x
        y = snake_head_y + 1
        for i in range(len(snake_body)):
            if snake_body[i] == (x,y):
                adjoining_body_bottom = 1
                break

        x = snake_head_x - 1
        y = snake_head_y
        for i in range(len(snake_body)):
            if snake_body[i] == (x,y):
                adjoining_body_left = 1
                break
            

        x = snake_head_x + 1
        y = snake_head_y
        for i in range(len(snake_body)):
            if snake_body[i] == (x,y):
                adjoining_body_right = 1
                break

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)