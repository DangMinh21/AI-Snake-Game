import pygame
import random
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    right = 1
    left = 2
    up = 3
    down = 4

Point = namedtuple('Point', 'x, y')

class Color(Enum):
    white = (255, 255, 255)
    red = (200,0,0)
    blue = (0, 0, 255)
    light_blue = (107, 107, 242)
    black = (0,0,0)
    gray = (100, 100, 100)

BLOCK_SIZE = 20
SPEED = 5


pygame.init()


class SnakeGame:
    def __init__(self, w: int  = 640, h: int = 480) -> None:
        # config of display
        self.w = w 
        self.h = h 
        self.wall = self.build_wall()
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        # config snake
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        self.direction = Direction.right

        
        self.score = 0
        self.food = self.create_food()

        self.game_over = False


    # create food randomly
    def create_food(self) -> Point:
        food = Point((random.randint(BLOCK_SIZE, self.w - BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE,
                        (random.randint(BLOCK_SIZE, self.h - BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
        if food in self.snake:
            self.create_food()
        return food
    
    def play_step(self):
        # 1. get event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_LEFT and self.direction != Direction.right:
                    self.direction = Direction.left

                elif event.key == pygame.K_RIGHT and self.direction != Direction.left:
                    self.direction = Direction.right

                elif event.key == pygame.K_UP and self.direction != Direction.down:
                    self.direction = Direction.up

                elif event.key == pygame.K_DOWN and self.direction != Direction.up:
                    self.direction = Direction.down

        # 2. move
        self.move(self.direction)
        self.snake.insert(0, self.head)

        # 3. check if game over
        if self.is_over():
            self.game_over = True
            return self.game_over, self.score
        
        
        # 4. snake eat the food or not
        if self.head == self.food:
            self.score += 1
            self.food = self.create_food()
        else:
            self.snake.pop()

        # 5. render new state in monitor
        self.render()
        self.clock.tick(SPEED)
        return self.game_over, self.score
        

    def move(self, direction) -> None:
        x = self.head.x
        y = self.head.y
        # update head of snake
        if direction == Direction.right:
            x += BLOCK_SIZE
        elif direction == Direction.left:
            x -= BLOCK_SIZE
        elif direction == Direction.down:
            y += BLOCK_SIZE
        elif direction == Direction.up:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
    
    def is_over(self) -> bool:
        if self.head.x < BLOCK_SIZE or self.head.x > self.w - BLOCK_SIZE - 1 or \
            self.head.y < BLOCK_SIZE or self.head.y > self.h - BLOCK_SIZE - 1:
            return True
        if self.head in self.snake[1:]:
            return True
        return False
    
    def render(self):
        self.display.fill('black')

        for point in self.snake:
            pygame.draw.rect(self.display, Color.blue.value, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, Color.light_blue.value, pygame.Rect(point.x + 4, point.y + 4, 12, 12))

        for point in self.wall:
            pygame.draw.rect(self.display, Color.gray.value, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, 'red', pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = pygame.font.SysFont('arial', 15).render(f"Score: {str(self.score)}", True, Color.white.value, )
        self.display.blit(text, dest=[0, 0])
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


def play():
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


if __name__ == "__main__":
    play()