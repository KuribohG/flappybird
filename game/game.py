import sys
import random

# import numpy as np
import pygame

pygame.init()

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
BASEY = SCREENHEIGHT * 0.79

FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

# def get_mask(img):
#     width = img.get_width()
#     height = img.get_height()
#     mask = np.zeros((width, height)).astype('bool')
#     for x in range(width):
#         for y in range(height):
#             mask[x, y] = bool(img.get_at((x, y))[3])
#     return mask

def load_resources():
    IMAGES, MASKS = {}, {}
    
    IMAGES['digits'] = tuple([
        pygame.image.load('game/resources/sprites/'+str(i)+'.png').convert_alpha()
            for i in range(10)])

    IMAGES['base'] = pygame.image.load('game/resources/sprites/base.png').convert_alpha()

    IMAGES['player'] = (
        pygame.image.load('game/resources/sprites/redbird-downflap.png').convert_alpha(), 
        pygame.image.load('game/resources/sprites/redbird-midflap.png').convert_alpha(), 
        pygame.image.load('game/resources/sprites/redbird-upflap.png').convert_alpha(), 
    )

    IMAGES['lower_pipe'] = pygame.image.load('game/resources/sprites/pipe-green.png').convert_alpha()
    IMAGES['upper_pipe'] = pygame.transform.rotate(IMAGES['lower_pipe'], 180)

    IMAGES['background'] = pygame.image.load('game/resources/sprites/background-black.png').convert_alpha()

#   for k, v in IMAGES.items():
#       if isinstance(v, tuple):
#           MASKS[k] = tuple(map(get_mask, v))
#       else:
#           MASKS[k] = get_mask(v)

    return IMAGES, MASKS

IMAGES, MASKS = load_resources()

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['lower_pipe'].get_width()
PIPE_HEIGHT = IMAGES['lower_pipe'].get_height()
PIPEGAPSIZE = 110

BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX = [0, 1, 2, 1]

def getRandomPipe():
    gapYs = list(range(60, 270, 30))
    gapY = gapYs[random.randint(0, len(gapYs) - 1)]
    
    pipeX = SCREENWIDTH + 10
    
    return {'x': pipeX, 'y': gapY - PIPE_HEIGHT}, \
           {'x': pipeX, 'y': gapY + PIPEGAPSIZE}

def checkCrash(player, upperPipes, lowerPipes):
    if player['y'] + PLAYER_WIDTH >= BASEY:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'], 
                PLAYER_WIDTH, PLAYER_HEIGHT)

        upperPipes = [pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
                for uPipe in upperPipes]
        lowerPipes = [pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
                for lPipe in lowerPipes]

        uCollide = playerRect.collidelist(upperPipes)
        lCollide = playerRect.collidelist(lowerPipes)

        if uCollide == -1 and lCollide == -1:
            return False
        else:
            return True

class GameState:
    def __init__(self):
        self.score = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.pipeVelX = -4
        self.playerVelY = 0
        self.playerMaxVelY = 13
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False
        self.basex = 0
        uPipe, lPipe = getRandomPipe()
        self.upperPipes = [uPipe]
        self.lowerPipes = [lPipe]
        self.frame = 0
        self.nextPipe = 0
        self.score = 0

    def show_score(self):
        score_str = str(self.score)
        l = len(score_str)
        digit_width = IMAGES['digits'][0].get_width()
        startx = (SCREENWIDTH - l * digit_width) // 2
        y = 30
        for digit in score_str:
            SCREEN.blit(IMAGES['digits'][int(digit)], (startx, y))
            startx += digit_width

    def frame_step(self, input_actions):
        reward = 0.1
        terminal = False

        pygame.event.pump()

        assert sum(input_actions) == 1, "Which action would you take?"

        if input_actions[1] == 1:
            self.playerFlapped = True
            self.playerVelY = self.playerFlapAcc

        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        
        if not self.playerFlapped:
            self.playerVelY = min(self.playerMaxVelY,
                    self.playerVelY + self.playerAccY)
        else:
            self.playerFlapped = False

        self.playery = max(self.playery + self.playerVelY, 0)
        
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery}, 
                self.upperPipes, self.lowerPipes)

        if isCrash:
            terminal = True
            reward = -1
            self.__init__()

        if 130 < self.upperPipes[self.nextPipe]['x'] < 135:
            uPipe, lPipe = getRandomPipe()
            self.upperPipes.append(uPipe)
            self.lowerPipes.append(lPipe)

        if playerMidPos > self.upperPipes[self.nextPipe]['x']:
            self.score += 1
            self.nextPipe += 1

        while self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
            self.nextPipe -= 1

        self.frame += 1

        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['upper_pipe'], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['lower_pipe'], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        SCREEN.blit(IMAGES['player'][PLAYER_INDEX[self.frame % 4]], 
                (self.playerx, self.playery))
        self.show_score()

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        img = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()
        FPSCLOCK.tick(FPS)
        
        return img, reward, terminal

def main():
    game_state = GameState()
    
    while True:
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_SPACE]:
            game_state.frame_step([0, 1])
        else:
            game_state.frame_step([1, 0])
    
if __name__ == '__main__':
    main()
