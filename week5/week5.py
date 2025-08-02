import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import pickle
import os
import time


pygame.init()


BLOCK_SIZE = 20
SPEED = 1000  
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class SnakeGameAI:
    def __init__(self, w=640, h=480, headless=False):
        self.w = w
        self.h = h
        self.headless = headless
        
        if not headless:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()
        
        self.reset()
        
    def reset(self):
        self.direction = RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            
    def play_step(self, action):
        self.frame_iteration += 1
        
        if not self.headless:
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
      
        self._move(action)
        self.snake.insert(0, self.head)
        
     
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
      
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
      
        if not self.headless:
            self._update_ui()
            self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
       
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
       
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == RIGHT:
            x += BLOCK_SIZE
        elif self.direction == LEFT:
            x -= BLOCK_SIZE
        elif self.direction == DOWN:
            y += BLOCK_SIZE
        elif self.direction == UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        
    def get_state(self, game):
        head = game.head
        food = game.food
        
       
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN
        
        state = (
            
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
           
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
          
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
           
            food.x < head.x, 
            food.x > head.x, 
            food.y < head.y,  
            food.y > head.y   
        )
        
        return state
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]
        
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_model(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)

class DQNNetwork(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, output_size=3):  
        super(DQNNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DQNAgent:
    def __init__(self, input_size=11, hidden_size=128, output_size=3, learning_rate=0.001, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=5000, batch_size=32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = 0.95
        
       
        self.q_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network = DQNNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
      
        self.memory = deque(maxlen=memory_size)
        
        
        self.update_target_network()
        
    def get_state(self, game):
        head = game.head
        food = game.food
        
       
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN
        
        state = np.array([
           
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
           
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
           
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
          
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
           
            food.x < head.x,  
            food.x > head.x,  
            food.y < head.y,  
            food.y > head.y   
        ], dtype=np.float32)
        
        return state
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)
    
    def load_model(self, filename):
        if os.path.exists(filename):
            self.q_network.load_state_dict(torch.load(filename))

def train_q_learning(episodes=500, headless=True):
    print("Training Q-Learning Agent...")
    game = SnakeGameAI(headless=headless)
    agent = QLearningAgent()
    
    scores = []
    mean_scores = []
    start_time = time.time()
    
    for episode in range(episodes):
        game.reset()
        state = agent.get_state(game)
        
        while True:
            action = agent.get_action(state)
            final_move = [0, 0, 0]
            final_move[action] = 1
            
            reward, game_over, score = game.play_step(final_move)
            next_state = agent.get_state(game)
            
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            
            if game_over:
                break
        
        scores.append(score)
        mean_scores.append(np.mean(scores[-100:]))
        
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            print(f'Episode {episode}, Score: {score}, Mean Score: {mean_scores[-1]:.2f}')
                
    agent.save_model('q_learning_model.pkl')
    return scores, mean_scores

def train_dqn(episodes=500, headless=True):
    print("Training DQN Agent...")
    game = SnakeGameAI(headless=headless)
    agent = DQNAgent()
    
    scores = []
    mean_scores = []
    start_time = time.time()
    
    for episode in range(episodes):
        game.reset()
        state = agent.get_state(game)
        
        while True:
            action = agent.get_action(state)
            final_move = [0, 0, 0]
            final_move[action] = 1
            
            reward, game_over, score = game.play_step(final_move)
            next_state = agent.get_state(game)
            
            agent.remember(state, action, reward, next_state, game_over)
            state = next_state
            
            if game_over:
                break
        
        agent.replay()
        
     
        if episode % 50 == 0:
            agent.update_target_network()
        
        scores.append(score)
        mean_scores.append(np.mean(scores[-100:]))
        
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            print(f'Episode {episode}, Score: {score}, Mean Score: {mean_scores[-1]:.2f}, '
                  f'Epsilon: {agent.epsilon:.3f}, Time: {elapsed:.1f}s')
    
    agent.save_model('dqn_model.pth')
    print(f"DQN training completed in {time.time() - start_time:.1f}s")
    return scores, mean_scores

def test_agent(agent_type='q_learning', episodes=50, headless=True):
    print(f"Testing {agent_type} Agent...")
    game = SnakeGameAI(headless=headless)
    
    if agent_type == 'q_learning':
        agent = QLearningAgent()
        agent.load_model('q_learning_model.pkl')
    else:
        agent = DQNAgent()
        agent.load_model('dqn_model.pth')
    
    agent.epsilon = 0
    scores = []
    
    for episode in range(episodes):
        game.reset()
        
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state)
            final_move = [0, 0, 0]
            final_move[action] = 1
            
            reward, game_over, score = game.play_step(final_move)
            
            if game_over:
                scores.append(score)
                break
    
    return scores

def plot_results(q_scores, q_mean_scores, dqn_scores, dqn_mean_scores):
    plt.figure(figsize=(15, 5))
    
  
    plt.subplot(1, 3, 1)
    plt.plot(q_mean_scores, label='Q-Learning', color='blue', linewidth=2)
    plt.plot(dqn_mean_scores, label='DQN', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Mean Score (last 100 episodes)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
   
    plt.subplot(1, 3, 2)
  
    window = 20
    if len(q_scores) > window:
        q_smooth = np.convolve(q_scores, np.ones(window)/window, mode='valid')
        dqn_smooth = np.convolve(dqn_scores, np.ones(window)/window, mode='valid')
        plt.plot(q_smooth, alpha=0.8, label='Q-Learning', color='blue')
        plt.plot(dqn_smooth, alpha=0.8, label='DQN', color='red')
    else:
        plt.plot(q_scores, alpha=0.8, label='Q-Learning', color='blue')
        plt.plot(dqn_scores, alpha=0.8, label='DQN', color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Score (smoothed)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

def play_with_agent(agent_type='dqn', episodes=3):
    print(f"Playing {episodes} games with {agent_type} agent...")
    game = SnakeGameAI(headless=False) 
    
    if agent_type == 'q_learning':
        agent = QLearningAgent()
        agent.load_model('q_learning_model.pkl')
    else:
        agent = DQNAgent()
        agent.load_model('dqn_model.pth')
    
    agent.epsilon = 0
    
    for episode in range(episodes):
        game.reset()
        print(f"Game {episode + 1}")
        
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state)
            final_move = [0, 0, 0]
            final_move[action] = 1
            
            reward, game_over, score = game.play_step(final_move)
            
            if game_over:
                print(f'Game {episode + 1} Score: {score}')
                time.sleep(1)  
                break
    
    pygame.quit()

def main():
  
     
    print("\n1. Training Q-Learning Agent...")
    q_scores, q_mean_scores = train_q_learning(episodes=500, headless=True)
    
    print("\n2. Training DQN Agent...")
    dqn_scores, dqn_mean_scores = train_dqn(episodes=500, headless=True)

 
    print("\n3. Generating comparison plots...")
    plot_results(q_scores, q_mean_scores, dqn_scores, dqn_mean_scores)
  
    print(f"Q-Learning - Final Mean Score: {q_mean_scores[-1]:.2f}")
    print(f"DQN - Final Mean Score: {dqn_mean_scores[-1]:.2f}")
    
    print("\n4. Testing trained agents...")
    q_test_scores = test_agent('q_learning', 50)
    dqn_test_scores = test_agent('dqn', 50)
    
    print(f"\nTEST PERFORMANCE (50 episodes):")
    print(f"Q-Learning - Mean: {np.mean(q_test_scores):.2f}, Std: {np.std(q_test_scores):.2f}")
    print(f"DQN - Mean: {np.mean(dqn_test_scores):.2f}, Std: {np.std(dqn_test_scores):.2f}")
   
if __name__ == "__main__":
    main()