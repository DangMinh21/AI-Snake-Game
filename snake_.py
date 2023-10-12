import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy

pygame.init()

class Direction(Enum):
    right = 1
    left = 2
    up = 3
    down = 4

class Action(Enum):
    straight = (1, 0, 0) # go straight forward
    right = (0, 1, 0) # turn left
    left = (0, 0, 1) # turn right

Point = namedtuple('Point', 'x, y')

class Color(Enum):
    white = (255, 255, 255)
    red = (200,0,0)
    blue = (0, 0, 255)
    light_blue = (107, 107, 242)
    black = (0,0,0)
    gray = (100, 100, 100)

BLOCK_SIZE = 20
SPEED = 60


class SnakeGame:
    def __init__(self, w: int  = 640, h: int = 480) -> None:
        # config of display
        self.w = w 
        self.h = h 
        self.wall = self.build_wall()
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # config snake
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        
        self.direction = Direction.right

        self.score = 0
        self.food = self.create_food()
        self.frame_iter = 0


    # create food randomly
    def create_food(self) -> Point:
        food = Point((random.randint(BLOCK_SIZE, self.w - BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE,
                        (random.randint(BLOCK_SIZE, self.h - BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
        if food in self.snake:
            self.create_food()
        return food
    

    def play_step(self, action):
        self.frame_iter += 1
        # 1. get event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self.move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_over() or self.frame_iter > 10000 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # 4. snake eat the food or not
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self.create_food()
        else:
            self.snake.pop()

        # 5. render new state in monitor
        self.render()
        self.clock.tick(SPEED)
        return reward, game_over, self.score


    def move(self, action: Action) -> None:
        
        clock_wise = [Direction.right, Direction.down, Direction.left, Direction.up]
        current_direction_index = clock_wise.index(self.direction)
        
        if action == Action.straight:
            new_direction = clock_wise[current_direction_index]
        
        elif action == Action.right: # clock wise 
            new_direction_index = (current_direction_index + 1) % 4 # 4 direction
            new_direction = clock_wise[new_direction_index]
        
        elif action == Action.left: # counter clock wise
            new_direction_index = (current_direction_index - 1) % 4 
            new_direction = clock_wise[new_direction_index]
        self.direction = new_direction

        
        # update head of snake
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.right:
            x += BLOCK_SIZE
        elif self.direction == Direction.left:
            x -= BLOCK_SIZE
        elif self.direction == Direction.down:
            y += BLOCK_SIZE
        elif self.direction == Direction.up:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

        

    def is_over(self, point=None) -> bool:
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE - 1 or point.x < BLOCK_SIZE or \
              point.y > self.h - BLOCK_SIZE - 1 or point.y < BLOCK_SIZE:
            return True
        # hits itself
        if point in self.snake[1:]:

            return True

        return False
    

    def render(self):
        self.display.fill(Color.black.value)

        for point in self.snake:
            pygame.draw.rect(self.display, Color.blue.value, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, Color.light_blue.value, pygame.Rect(point.x + 4, point.y + 4, 12, 12))

        for point in self.wall:
            pygame.draw.rect(self.display, Color.gray.value, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, 'red', pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = pygame.font.SysFont('arial', 15).render(f"Score: {str(self.score)}", True, Color.white.value, )
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def build_wall(self):
        """
        ==================
        ||              ||
        ||              ||
        ||              ||
        ==================
        """
        wall = []
        wall_up = [Point(x * BLOCK_SIZE, 0) for x in range(0, self.w // BLOCK_SIZE)]
        wall_down = [Point(x * BLOCK_SIZE, (self.h // BLOCK_SIZE - 1) * BLOCK_SIZE )
                      for x in range(0, self.w // BLOCK_SIZE)]
        wall_left = [Point(0, y * BLOCK_SIZE) for y in range(0, self.h // BLOCK_SIZE)]
        wall_right = [Point((self.w // BLOCK_SIZE - 1) * BLOCK_SIZE, y * BLOCK_SIZE) 
                      for y in range(0, self.h // BLOCK_SIZE)]
        wall.extend(wall_up)
        wall.extend(wall_down)
        wall.extend(wall_left)
        wall.extend(wall_right)
        return wall
    


# if __name__ == "__main__":
#     game = SnakeGame()
#     while True:
#         _, done, _ = game.play_step(Action.straight)
#         if done:
#             break

