import pygame

from snake import SnakeGame
from snake_ import SnakeGame as SnakeGameAI
from agent import Agent

def human_play():
    # create a snake game object
    game = SnakeGame()

    # play snake game
    while(True):
        game_over, score = game.play_step()

        if game_over == True:
            break

    print(f"Score: {score}")

    # quit game
    pygame.quit()


def agent_play():
    # 1. create game
    game = SnakeGameAI()

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


def main():
    who = ''

    if who == "agent":
         agent_play()

    else: 
         human_play()


if __name__ == '__main__':
    main()