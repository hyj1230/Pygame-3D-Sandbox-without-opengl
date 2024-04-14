import pygame

screen = pygame.display.set_mode((600, 600), pygame.RESIZABLE)
pygame.init()

import sys
import numpy as np
import model

clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 11)

window = model.Window(screen)

while 1:
    FPS = str(round(clock.get_fps()))
    pygame.display.set_caption('fps:' + FPS)
    pygame_events = pygame.event.get()
    for event in pygame_events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((round(0.5*255), round(0.69*255), round(1.0*255)))
    z_buffer = np.full(screen.get_size(), np.inf, dtype=np.float64)
    window.handle_event(pygame_events)
    window.update()
    window.on_draw(z_buffer)

    pygame.display.flip()
    clock.tick(114514)
