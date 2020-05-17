import colorsys
import math
import random
from rtree import index
import torchvision
import os
import numpy as np
import cv2

def random_color():
    hue = random.random()
    lightness = random.random() * 0.3 + 0.5
    saturation = random.random() * 0.2 + 0.7
    return tuple(map(lambda f: int(f * 255), colorsys.hls_to_rgb(hue, lightness, saturation)))

    #colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    #color = ""
    #for i in range(6):
     #   color += colorArr[random.randint(0, 14)]
    #return "#" + color

class Creature:
    """Creature class representing one bouncing ball for now."""
    id_count = 0

    def __init__(self, x, y, dx, dy, color, radius):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.radius = radius
        self.id = Creature.id_count
        Creature.id_count += 1
        self.collision_count = 0

    def step(self, world, overlapping):

        self.x += self.dx
        self.y += self.dy

        ddx = random.random() * 0.2 - 0.15
        ddy = random.random() * 0.2 - 0.15

        if self.collision_count >1:
            self.color = random_color()
            self.collision_count = 0

        # Bounce off walls:
        bounding = -1
        if self.x < self.radius + bounding:
            ddx -= self.x - self.radius
            self.collision_count +=1
        elif self.x > world.width - self.radius -bounding:
            ddx += world.width - self.radius - self.x
            self.collision_count += 1
        if self.y < self.radius +bounding :
            ddy -= self.y - self.radius
            self.collision_count += 1
        elif self.y > world.height - self.radius -bounding:
            ddy += world.height - self.radius - self.y
            self.collision_count += 1


        for other in overlapping:
            dist = self.distance(other.x, other.y)
            if dist:
                ddx -= 3 * (other.x - self.x) / dist
                ddy -= 3 * (other.y - self.y) / dist

        self.dx += ddx
        self.dy += ddy

        # Maximum speed:
        speed = 8
        if abs(self.dx) > speed:
            self.dx *= speed / abs(self.dx)
        if abs(self.dy) > speed:
            self.dy *= speed / abs(self.dy)




    def draw(self, screen):
        #pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.radius))
        cv2.circle(screen, (int(self.x), int(self.y)), int(self.radius), self.color, -1)

    def box(self):
        """Return the bounding box for this creature"""
        r2 = self.radius
        res = [self.x - r2, self.y - r2, self.x + r2, self.y + r2]
        return res

    def distance(self, x1, y1):
        dx = self.x - x1
        dy = self.y - y1
        return math.sqrt(dx * dx + dy * dy)


class World:
    """The world and the creatures in it. Also has an r-tree for collision detection."""

    def __init__(self):
        self.index = index.Index()
        self.width = 64
        self.height = 64
        self.creatures = {}

    def add_creature(self, creature):
        self.creatures[creature.id] = creature

    def del_creature(self, creature):
        Creature.id_count -= 1
        del self.creatures[creature]

    def step(self):
        for creature in self.creatures.values():
            self.index.delete(creature.id, creature.box())
            creature.step(self, self.overlaps(creature))
            self.index.add(creature.id, creature.box())

    def draw(self, screen):
        for creature in self.creatures.values():
            creature.draw(screen)

    def random_creature(self):
        radius = 6
        x = self.width // 2 - 2 * radius * np.random.normal(0, 1)
        if x>=self.width or x<=0:
            x = random.random() * self.width
        y = self.height // 2 + 5 * radius * np.random.normal(0, 1)
        if y >= self.width or y <= 0:
            y = random.random() * self.height
        creature = Creature(x,y,
                            random.random() * 6 - 4,
                            random.random() * 4 - 4,
                            color=random_color(),
                            radius=radius)
        return creature

    def overlaps(self, creature):
        """Return any creature inside the circle (x,y) with the given radius."""
        res = []
        for candidate_id in self.index.intersection(creature.box()):
            candidate = self.creatures[candidate_id]
            if creature.distance(candidate.x, candidate.y) < candidate.radius + creature.radius:
                res.append(candidate)
        return res

class BouncingBalls_gen():
    def __init__(self, n_frames_total, n_balls):
        super(BouncingBalls_gen, self).__init__()
        self.world = World()
        self.frames = n_frames_total
        self.n_balls = n_balls

    def gen_ball_seq(self):
        dat = np.zeros((self.frames, self.world.width, self.world.height, 3), dtype=np.float32)
        for _ in range(self.n_balls):
            self.world.add_creature(self.world.random_creature())

        for i in range(self.frames):
            self.world.step()
            self.world.draw(dat[i])

        dat = 1 - dat.transpose((0,3,1,2))/255
        self.world.del_creature(0)
        self.world.del_creature(1)
        dat = 2 * (dat-0.5)
        return dat

path = './dataset/'

if not os.path.exists(path):
    os.makedirs(path)
np.random.seed(0)
a = BouncingBalls_gen(20, 2)
for i in range(6687):
    aa = a.gen_ball_seq()
    np.save('%s/%d'%(path, i), aa)
