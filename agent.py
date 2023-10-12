import torch
import random
import numpy as np
from collections import deque
from snake_ import SnakeGame, Direction, Point, Action, BLOCK_SIZE
from models import LinearQNet, QNetTrainer
from utilis import plot


MEMORY_SIZE = 100000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
RANDOM_FIX = 5

class Agent:
    def __init__(self, retrain = False, file_path='') -> None:
        self.n_games = 0
        self.epsilon = 0
        self.retrain = retrain
        self.file_path = file_path
        self.memory = deque(maxlen = MEMORY_SIZE)
        self.model = self.load_model()
        self.trainer = QNetTrainer(model=self.model, lr=LR, gamma=GAMMA)

    def load_model(self):
        model = LinearQNet(11, 256, 3)
        if self.retrain:
            print(f'LOAD MODEL')
            model.load_state_dict(torch.load(self.file_path))
            model.train()
        return model

    def get_state(self, game: SnakeGame):
        """
        return current vector state of game:
            [danger_straight, right_danger, left_danger, 
            direction_up, direction_down,direction_right, direction_left,
            food_right, food_left, food_up, food_down]
        """
        # current direction of snake: [dir_u, dir_d, dir_r, dir_l], datatype: boolean
        dir_u = game.direction == Direction.up
        dir_d = game.direction == Direction.down
        dir_r = game.direction == Direction.right
        dir_l = game.direction == Direction.left
        direction = np.array([dir_u, dir_d, dir_r, dir_l])

        # points around head of snake
        head = game.snake[0]
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_l = Point(head.x - BLOCK_SIZE, head.y)


        # if move straight, check game over
        straight_danger = (dir_u and game.is_over(point_u)) or \
                          (dir_d and game.is_over(point_d)) or \
                          (dir_r and game.is_over(point_r)) or \
                          (dir_l and game.is_over(point_l))
        
        # if move right, check game over
        right_danger = (dir_u and game.is_over(point_r)) or \
                       (dir_d and game.is_over(point_l)) or \
                       (dir_r and game.is_over(point_d)) or \
                       (dir_l and game.is_over(point_u))
        
        # if move left, check game over
        left_danger = (dir_u and game.is_over(point_l)) or \
                      (dir_d and game.is_over(point_r)) or \
                      (dir_r and game.is_over(point_u)) or \
                      (dir_l and game.is_over(point_d))
        danger = np.array([straight_danger, right_danger, left_danger])
        
        # find food location represent by a vector [food_up, food_down, food, r, food_l]
        food_u = head.y > game.food.y
        food_d = head.y < game.food.y
        food_r = head.x < game.food.x
        food_l = head.x > game.food.x
        food_location = np.array([food_u, food_d, food_r, food_l])

        return np.concatenate((danger, direction, food_location), axis=None)

    # get the action that agent choose: trade off between exploration and exploitation
    def get_action(self, state):
        actions = ['straight', 'right', 'left']
        self.epsilon = 80 - self.n_games    # more game, less random choice

        if not self.retrain:
            if random.randint(0, 200) < self.epsilon:
                move  = random.randint(0, 2)
                return Action[actions[move]]
            else:
                state = torch.tensor(state, dtype=torch.float)
                predicted_move = self.model(state)
                move = predicted_move.argmax().item()
                return Action[actions[move]]
            
        state = torch.tensor(state, dtype=torch.float)
        predicted_move = self.model(state)
        move = predicted_move.argmax().item()
        return Action[actions[move]]
        


        # if self.n_games < 80:
        #     if random.randint(0, 200) < self.epsilon:
        #         move  = random.randint(0, 2)
        #         return Action[actions[move]]
        #     else:
        #         state = torch.tensor(state, dtype=torch.float)
        #         predicted_move = self.model(state)
        #         move = predicted_move.argmax().item()
        #         return Action[actions[move]]

        # else:
        #     if random.randint(0, 100) < RANDOM_FIX:
        #         move  = random.randint(0, 2)
        #         return Action[actions[move]]
        #     else:
        #         state = torch.tensor(state, dtype=torch.float)
        #         predicted_move = self.model(state)
        #         move = predicted_move.argmax().item()
        #         return Action[actions[move]]


    def remember(self, current_state, action, next_state, reward, done):
        self.memory.append((current_state, action, next_state, reward, done))

    def train_short_memory(self, current_state, action, next_state, reward, done):
        self.trainer.train(current_state, action, next_state, reward, done)

    def train_long_menmory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        else:
            mini_sample = self.memory

        current_states, actions, next_states, rewards, dones = zip(*mini_sample)
        self.trainer.train(current_states, actions, next_states, rewards, dones)

def train(retrain = False):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0

        model_path = 'model/model.pth'
        record = 0
        agent = Agent(retrain, file_path=model_path)
        game = SnakeGame()

        while True:
            # 1. get current state
            current_state = agent.get_state(game)

            # 2. get action
            action = agent.get_action(current_state)

            # 3. get next state
            reward, done, score = game.play_step(action)
            next_state = agent.get_state(game)

            # 4. agent learn
            agent.train_short_memory(current_state, action.value, next_state, reward, done)

            agent.remember(current_state, action.value, next_state, reward, done)

            if done:
                # train long memory and reset game
                agent.n_games += 1
                agent.train_long_menmory()

                if score > record:
                    record = score
                    agent.model.save()

                print(f"Game: {agent.n_games} - Score: {score} - Recore: {record}")

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                game.reset()
                

def Agent_play_snake():
    # 1. create game
    game = SnakeGame()

    # 2. Agent
    agent = Agent(retrain=True, file_path='model/model.pth')
    while True:
            # 1. get current state
            current_state = agent.get_state(game)

            # 2. get action
            action = agent.get_action(current_state)

            # 3. get next state
            _, done, score = game.play_step(action)

            print(f"Game: {agent.n_games} - Score: {score}")

            if done:
                break


if __name__ == "__main__":
    Agent_play_snake()
    # train(retrain=True)
    # game = SnakeGame()
    # while True:
    #     _, done, _ = game.play_step(Action.straight)
    #     if done:
    #         break


    
