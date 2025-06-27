import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GAME_WIDTH = 600
GAME_HEIGHT = 600
CELL_SIZE = 20
CELL_NUMBER = GAME_WIDTH // CELL_SIZE

# Basic Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Snake:
    def __init__(self):
        self.body = [pygame.Vector2(5, 10), pygame.Vector2(4, 10), pygame.Vector2(3, 10)]
        self.direction = pygame.Vector2(1, 0)
        self.new_block = False

    def draw_snake(self, screen):
        for block in self.body:
            x = int(block.x * CELL_SIZE)
            y = int(block.y * CELL_SIZE)
            pygame.draw.rect(screen, GREEN, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

    def move_snake(self):
        if self.new_block:
            self.body.insert(0, self.body[0] + self.direction)
            self.new_block = False
        else:
            self.body = [self.body[0] + self.direction] + self.body[:-1]

    def add_block(self):
        self.new_block = True

    def check_collision(self):
        head = self.body[0]
        if not 0 <= head.x < CELL_NUMBER or not 0 <= head.y < CELL_NUMBER:
            return True
        return head in self.body[1:]

class Food:
    def __init__(self):
        self.randomize()

    def draw_food(self, screen):
        x = int(self.pos.x * CELL_SIZE)
        y = int(self.pos.y * CELL_SIZE)
        pygame.draw.rect(screen, RED, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

    def randomize(self):
        self.pos = pygame.Vector2(random.randint(0, CELL_NUMBER - 1), random.randint(0, CELL_NUMBER - 1))

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.high_score = self.load_high_score()
        self.game_over = False
        self.game_started = False

    def update(self):
        if not self.game_over and self.game_started:
            self.snake.move_snake()
            self.check_collision()
            self.check_fail()

    def draw_elements(self, screen):
        screen.fill(BLACK)
        self.food.draw_food(screen)
        self.snake.draw_snake(screen)

    def check_collision(self):
        if self.food.pos == self.snake.body[0]:
            self.food.randomize()
            self.snake.add_block()
            self.score += 1
            while self.food.pos in self.snake.body:
                self.food.randomize()

    def check_fail(self):
        if self.snake.check_collision():
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()

    def restart_game(self):
        self.__init__()

    def start_game(self):
        self.game_started = True
        self.game_over = False

    def load_high_score(self):
        try:
            with open('high_score.txt', 'r') as f:
                return int(f.read())
        except:
            return 0

    def save_high_score(self):
        with open('high_score.txt', 'w') as f:
            f.write(str(self.high_score))

def draw_text(screen, text, size, x, y, color=WHITE):
    font = pygame.font.Font(None, size)
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y))
    screen.blit(surface, rect)
    return rect

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()

    game = Game()
    SCREEN_UPDATE = pygame.USEREVENT
    pygame.time.set_timer(SCREEN_UPDATE, 150)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == SCREEN_UPDATE:
                game.update()

            if event.type == pygame.KEYDOWN:
                if game.game_started and not game.game_over:
                    if event.key == pygame.K_UP and game.snake.direction.y != 1:
                        game.snake.direction = pygame.Vector2(0, -1)
                    if event.key == pygame.K_DOWN and game.snake.direction.y != -1:
                        game.snake.direction = pygame.Vector2(0, 1)
                    if event.key == pygame.K_RIGHT and game.snake.direction.x != -1:
                        game.snake.direction = pygame.Vector2(1, 0)
                    if event.key == pygame.K_LEFT and game.snake.direction.x != 1:
                        game.snake.direction = pygame.Vector2(-1, 0)

                if event.key == pygame.K_SPACE and not game.game_started:
                    game.start_game()

                if event.key == pygame.K_r and game.game_over:
                    game.restart_game()

        screen.fill(BLACK)
        game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        game.draw_elements(game_surface)
        screen.blit(game_surface, ((WINDOW_WIDTH - GAME_WIDTH) // 2, (WINDOW_HEIGHT - GAME_HEIGHT) // 2))

        draw_text(screen, "SNAKE GAME", 48, WINDOW_WIDTH // 2, 30)
        draw_text(screen, f"Score: {game.score}", 28, 100, WINDOW_HEIGHT - 40)
        draw_text(screen, f"High Score: {game.high_score}", 28, WINDOW_WIDTH - 150, WINDOW_HEIGHT - 40)

        if not game.game_started and not game.game_over:
            draw_text(screen, "Press SPACE to Start", 36, WINDOW_WIDTH // 2, 150)

        if game.game_over:
            draw_text(screen, "GAME OVER", 48, WINDOW_WIDTH // 2, 150, RED)
            draw_text(screen, "Press R to Restart", 28, WINDOW_WIDTH // 2, 200)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
