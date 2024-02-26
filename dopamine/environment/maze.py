import gym
import cv2
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from PIL import Image, ImageDraw
import types
import cv2
import gin.tf

render_size = 256
obs_size = 84

class GameMap:
    def __init__(self, start_pos, reward_colors, reward_scales, reward_positions,
                 hpath_matrix, vpath_matrix, unused_matrix, transition_map=None):
        self.start_pos = start_pos
        self.reward_num = len(reward_colors)
        assert len(reward_colors) == self.reward_num
        assert len(reward_scales) == self.reward_num
        assert len(reward_positions) == self.reward_num

        h, w = np.array(hpath_matrix).shape
        self.height, self.width = h - 1, w - 1

        self.reward_colors = reward_colors
        self.reward_scales = reward_scales
        self.reward_positions = reward_positions
        self.hpath_matrix = hpath_matrix
        self.vpath_matrix = vpath_matrix
        self.unused_matrix = unused_matrix
        if transition_map is None:
            self.transition_map = {}
        else:
            self.transition_map = transition_map

def convert_to_matrix(shape, arr):
    matrix = np.zeros(shape)
    for v in arr:
        if len(v) == 2:
            matrix[v] = 1
        else:
            *v, direction = v
            matrix[tuple(v)] = direction
    return matrix

start_pos = (2, 1)

# color_set: the color in observation for each type of reward
# reward_scales: the reward function for each type of reward
# reward_positions: positions for each type of reward
# hpaths: if (i, j) is in hpaths, then the block (i, j) (with i-th rows, j-th column) can move to the left direction
# vpaths: if (i, j) is in vpaths, then the block (i, j) can move to the up direction
# (i, j, 2) or (i, j, 3) is one-way paths.

color_set = [(255, 200, 150), (150, 200, 255), (175, 255, 175), (255, 175, 175)]
reward_scales = [lambda: 0.2 * np.random.rand() + 0.2,
                 lambda: 0.2 * np.random.rand() + 0.8,
                 lambda: 0.2 * np.random.rand() + 0.3,
                 lambda: 0.2 * np.random.rand() + 0.5]
reward_positions = [[(0, 0)], [(0, 2)], [(2, 3)], [(3, 2)]]
hpaths = [(0, 0), (1, 3), (1, 4), (2, 3), (3, 2), (4, 2), (5, 2)]
vpaths = [(0, 1), (0, 2), (0, 3), (0, 5), (1, 4), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2)]
unused = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 4), (3, 3), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)]
hpath_matrix = convert_to_matrix((6, 6), hpaths)
vpath_matrix = convert_to_matrix((6, 6), vpaths)
unused_matrix = convert_to_matrix((5, 5), unused)

game_map = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix)

#  ================= map_exclusive ========================

start_pos = (0, 0)
color_set = [(175, 255, 175), (255, 175, 175)]
reward_scales = [lambda: 0.4 * np.random.rand() + 0.2,
                 lambda: 0.5 * np.random.rand() + 0.4]
reward_positions = [[(0, 1), (1, 2), (2, 3), (3, 4)], [(1, 0), (2, 1), (3, 2), (4, 3)]]
hpaths = [(1, 0), (1, 1, 3), (2, 1), (2, 2, 3), (3, 2), (3, 3, 3), (4, 3), (4, 4, 3)]
vpaths = [(0, 1), (1, 1, 3), (1, 2), (2, 2, 3), (2, 3), (3, 3, 3), (3, 4), (4, 4, 3), (4, 5)]
unused = [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)]

hpath_matrix = convert_to_matrix((6, 6), hpaths)
vpath_matrix = convert_to_matrix((6, 6), vpaths)
unused_matrix = convert_to_matrix((5, 5), unused)

game_map_exclusive = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix)

#  ================= map_inclusive ========================

start_pos = (0, 0)
color_set = [(175, 255, 175), (255, 175, 175)]
reward_scales = [lambda: 0.4 * np.random.rand() + 0.2,
                 lambda: 0.5 * np.random.rand() + 0.4]
reward_positions = [[(0, 1), (1, 4), (4, 2)], [(0, 2), (2, 4), (4, 1)]]
hpaths = [(1, 0), (1, 3, 3), (2, 3), (2, 4), (3, 3, 3), (3, 4), (4, 3), (4, 0), (5, 0)]
vpaths = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3, 3), (1, 4), (3, 0), (3, 1, 2), (3, 2), (3, 3), (3, 4, 2),
          (4, 1, 2), (4, 2), (4, 3)]
unused = [(0, 4), (2, 0), (2, 1), (2, 2), (4, 4)]

hpath_matrix = convert_to_matrix((6, 6), hpaths)
vpath_matrix = convert_to_matrix((6, 6), vpaths)
unused_matrix = convert_to_matrix((5, 5), unused)

game_map_inclusive = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix)

# ================= map_constraint ========================

start_pos = (0, 0)
color_set = [(255, 175, 175), (175, 255, 175), (150, 200, 255)]
reward_scales = [lambda: 1.0,
                 lambda: 1.0,
                 lambda: 1.0]
reward_positions = [[(3, 1), (3, 4)],
                    [(0, 3), (2, 1), (3, 2)],
                    [(2, 0), (4, 0), (4, 1), (4, 4)]]
unused = []
hpaths = [(1, 0, 3), (1, 2), (1, 4, 3), (2, 0), (2, 1), (2, 4),
          (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
          (4, 0), (4, 1), (4, 2), (4, 3, 3), (4, 4, 3),
          (5, 0), (5, 1), (5, 2), (5, 3)]
vpaths = [(0, 1, 3), (0, 2), (0, 3), (0, 4), (1, 1), (1, 3), (1, 4, 3),
          (2, 2), (2, 4, 2), (4, 4), (4, 5)]

hpath_matrix = convert_to_matrix((6, 6), hpaths)
vpath_matrix = convert_to_matrix((6, 6), vpaths)
unused_matrix = convert_to_matrix((5, 5), unused)

transition_map = {(1, 0) : [0, 0, 1/3, 2/3], (2, 1) : [0, 0, 1/2, 1/2]}

game_map_constraint = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix, transition_map)

# ================= map_constraint_2 ========================

start_pos = (0, 4)
color_set = [(255, 175, 175), (175, 255, 175), (150, 200, 255)]
reward_scales = [lambda: 1.0,
                 lambda: 1.0,
                 lambda: 1.0]
reward_positions = [[(2, 0), (2, 3), (2, 5), (4, 0), (4, 3), (4, 5), (6, 3)],
                    [(1, 2), (1, 4), (3, 4), (5, 1), (5, 4)],
                    [(1, 6), (5, 6), (3, 1)]]
unused = []
hpaths = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
          (2, 0, 3), (2, 1, 3), (2, 2, 3), (2, 3, 3), (2, 4, 3), (2, 5, 3), (2, 6, 3),
          (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
          (4, 0, 3), (4, 1, 3), (4, 2, 3), (4, 3, 3), (4, 4, 3), (4, 5, 3), (4, 6, 3),
          (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
          (6, 0, 3), (6, 1, 3), (6, 2, 3), (6, 3, 3), (6, 4, 3), (6, 5, 3), (6, 6, 3),
          (7, 0), (7, 6)
          ]
vpaths = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
          (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
          (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
          (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]

hpath_matrix = convert_to_matrix((8, 8), hpaths)
vpath_matrix = convert_to_matrix((8, 8), vpaths)
unused_matrix = convert_to_matrix((7, 7), unused)

game_map_constraint_2 = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix)

# ================= map_constraint_3 ========================

start_pos = (0, 0)
color_set = [(255, 175, 175), (175, 255, 175), (150, 200, 255)]
reward_scales = [lambda: 1.0,
                 lambda: 1.0,
                 lambda: 1.0]
reward_positions = [[(3, 1), (4, 4), (1, 4)],
                    [(0, 3), (2, 1), (1, 1)],
                    [(3, 0), (4, 0), (4, 1), (5, 1), (3, 4), (4, 3)]]
unused = [(2, 2), (3, 2), (4, 2), (5, 2)]
hpaths = [(1, 0, 3), (1, 2), (1, 4, 3), (1, 5, 3), (2, 0), (2, 1), (2, 5),
          (3, 0), (3, 1), (3, 2), (3, 3, 3), (3, 4, 3), (3, 5, 3),
          (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
          (5, 0), (5, 1), (5, 2), (5, 3, 3), (5, 4, 3), (5, 5, 3),
          (6, 0), (6, 1), (6, 3)]
vpaths = [(0, 1, 3), (0, 2), (0, 3), (0, 4), (0, 5), (1, 1), (1, 3), (1, 4, 3), (1, 5),
          (2, 4), (2, 5), (5, 4), (5, 5), (5, 6)]

hpath_matrix = convert_to_matrix((7, 7), hpaths)
vpath_matrix = convert_to_matrix((7, 7), vpaths)
unused_matrix = convert_to_matrix((6, 6), unused)

transition_map = {(1, 0) : [0, 0, 1/2, 1/2]}

game_map_constraint_3 = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix, transition_map)

# ================= map_constraint_v6 ========================

start_pos = (0, 0)
color_set = [(255, 175, 175), (175, 255, 175), (150, 200, 255)]
reward_scales = [lambda: 1.0,
                 lambda: 1.0,
                 lambda: 1.0]
reward_positions = [[(3, 0), (3, 5), (4, 1), (4, 4), (1, 4)],
                    [(2, 1), (3, 3), (4, 2), (4, 5)],
                    [(4, 0), (3, 4), (4, 3), (5, 2)]]
unused = []
hpaths = [(0, 1), (0, 5), (1, 0, 3), (1, 2), (2, 0), (2, 1), (2, 5),
          (3, 0), (3, 1), (3, 2), (3, 3, 3), (3, 4, 3), (3, 5, 3),
          (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
          (5, 0), (5, 1), (5, 2), (5, 3, 3), (5, 4, 3), (5, 5, 3),
          (6, 0), (6, 1), (6, 2), (6, 3)]
vpaths = [(0, 1, 3), (0, 2), (0, 3), (0, 4), (0, 5), (1, 1), (1, 3), (1, 4), (1, 5),
          (2, 2), (2, 4), (2, 5), (5, 4), (5, 5), (5, 6)]

hpath_matrix = convert_to_matrix((7, 7), hpaths)
vpath_matrix = convert_to_matrix((7, 7), vpaths)
unused_matrix = convert_to_matrix((6, 6), unused)

transition_map = {(1, 0) : [0, 0, 1/2, 1/2], (0, 1): [1/2, 0, 0, 1/2]}

game_map_constraint_v6 = GameMap(start_pos, color_set, reward_scales, reward_positions,
          hpath_matrix, vpath_matrix, unused_matrix, transition_map)

all_game_maps = [game_map, game_map_exclusive, game_map_inclusive,
                 game_map_constraint, game_map_constraint_2, game_map_constraint_3,
                 game_map_constraint_v6]



@gin.configurable
class GridWorld(gym.Env):
    def __init__(self, game_map: GameMap,
                 render_w=render_size, render_h=render_size, obs_w=obs_size, obs_h=obs_size, reward_scale=1.0,
                 constrain_env=False, constrain_value=None, constrain_type=None):
        print(f'==== reward_scale: {reward_scale} ====')
        self.reward_scale = reward_scale
        self.max_steps = 150
        self.game_map = copy.deepcopy(game_map)
        self.reward_dim = len(self.game_map.reward_scales)
        self.constrain_env = constrain_env
        self.constrain_value = constrain_value
        self.constrain_type = constrain_type

        self.agent_color = (100, 100, 100)
        self.width, self.height = game_map.width, game_map.height
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_w, obs_h, 3), dtype=np.uint8)
        self.delta = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.np_random = np.random
        self.get_randint = lambda x: self.np_random.randint(0, x)
        self.get_randpos = lambda: (self.get_randint(self.width), self.get_randint(self.height))

        self.obs_w, self.obs_h = obs_w, obs_h
        self.render_w, self.render_h = render_w, render_h

        def merge(a, b):
            ax, ay = a
            bx, by = b
            return (ax + bx, ay + by)
        def valid(a):
            ax, ay = a
            return 0 <= ax < self.width and 0 <= ay < self.height

        self._merge = merge
        self._valid = valid

    def get_movable_directions(self):
        movable_directions = []
        h, w = self.pos
        for idx, direction in enumerate(self.delta):
            movable = False
            if direction == (0, 1) and self.game_map.vpath_matrix[h, w + 1] in {1, 3}:
                movable = True
            if direction == (0, -1) and self.game_map.vpath_matrix[h, w] in {1, 2}:
                movable = True
            if direction == (1, 0) and self.game_map.hpath_matrix[h + 1, w] in {1, 3}:
                movable = True
            if direction == (-1, 0) and self.game_map.hpath_matrix[h, w] in {1, 2}:
                movable = True
            new_pos = self._merge(self.pos, direction)
            if self._valid(new_pos) and self.invalid_map[new_pos] > 0:
                movable = False
            if movable:
                movable_directions.append(idx)
        return movable_directions

    def reset(self):
        self.game_over = False
        self.current_steps = 0
        self.pos = self.game_map.start_pos
        self.target_matrix = np.zeros((self.height, self.width), np.int32)
        for idx, all_pos in enumerate(self.game_map.reward_positions):
            for pos in all_pos:
                self.target_matrix[pos] = idx + 1
        self.invalid_map = np.array(self.game_map.unused_matrix)
        return self.get_obs()

    def step(self, action):
        if self.pos in self.game_map.transition_map:
            p = self.game_map.transition_map[self.pos]
            action = np.random.choice(4, p=p)
        self.current_steps += 1

        reward = np.zeros((len(self.game_map.reward_scales, )))
        info = {}

        movable_directions = self.get_movable_directions()

        self.new_pos = self.simulate(self.pos, action)
        if not self._valid(self.new_pos) and action in movable_directions:
            agent_exit = True
        else:
            agent_exit = False

        if agent_exit:
            self.game_over = True
            return self.get_obs(), reward * self.reward_scale, True, info

        episode_end = False
        if self.current_steps == self.max_steps:
            episode_end = True
            self.game_over = True

        if action in movable_directions:
            if self.new_pos != self.pos:
                self.invalid_map[self.pos] = 1.0
                if self.target_matrix[self.new_pos] > 0:
                    index = self.target_matrix[self.new_pos] - 1
                    reward_scale = self.game_map.reward_scales[index]
                    if type(reward_scale) is types.FunctionType:
                        reward_scale = reward_scale()
                    reward[index] += reward_scale
                    self.target_matrix[self.new_pos] = 0
            self.pos = self.new_pos

        return self.get_obs(), reward * self.reward_scale, episode_end, info

    def render(self):
        image = self.get_obs()
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.savefig('test.png')

        plt.ion()
        plt.show()
        plt.pause(0.001)

    def compute_boundary(self, h, w):
        ws = 0.9 * self.render_w / self.width * w + 0.05 * self.render_w
        we = 0.9 * self.render_w / self.width * (w + 1) + 0.05 * self.render_w
        hs = 0.9 * self.render_h / self.height * h + 0.05 * self.render_h
        he = 0.9 * self.render_h / self.height * (h + 1) + 0.05 * self.render_h
        return ws, we, hs, he

    def get_obs(self):
        obs = np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8)
        obs[:, :] = (255, 255, 255)

        image = Image.fromarray(obs)
        draw = ImageDraw.Draw(image)

        sz = 2.0 * render_size / 256.0
        game_map = self.game_map
        # draw walls & unused block
        for h in range(self.height):
            for w in range(self.width):
                # draw top & left wall
                ws, we, hs, he = self.compute_boundary(h, w)
                if self.invalid_map[h, w] > 0:
                    draw.rectangle([(ws - sz, hs - sz), (we + sz, he + sz)], fill=(200, 200, 200))
                if game_map.hpath_matrix[h, w] in {0, 2, 3}:
                    color = {0: (0, 0, 0),
                             2: (255, 0, 0),
                             3: (255, 255, 0)}[game_map.hpath_matrix[h, w]]
                    draw.rectangle([(ws - sz, hs - sz), (we + sz, hs + sz)], fill=color)
                if game_map.vpath_matrix[h, w] in {0, 2, 3}:
                    color = {0: (0, 0, 0),
                             2: (255, 0, 0),
                             3: (255, 255, 0)}[game_map.vpath_matrix[h, w]]
                    draw.rectangle([(ws - sz, hs - sz), (ws + sz, he + sz)], fill=color)
                if w == self.width - 1:
                    # draw right wall
                    if game_map.vpath_matrix[h, w+1] in {0, 2, 3}:
                        color = {0: (0, 0, 0),
                                 2: (255, 0, 0),
                                 3: (255, 255, 0)}[game_map.vpath_matrix[h, w+1]]
                        draw.rectangle([(we - sz, hs - sz), (we + sz, he + sz)], fill=color)
                if h == self.height - 1:
                    # draw bottom wall
                    if game_map.hpath_matrix[h+1, w] in {0, 2, 3}:
                        color = {0: (0, 0, 0),
                                 2: (255, 0, 0),
                                 3: (255, 255, 0)}[game_map.hpath_matrix[h+1, w]]
                        draw.rectangle([(ws - sz, he - sz), (we + sz, he + sz)], fill=color)

        # draw agent
        agent_h, agent_w = self.pos
        ws, we, hs, he = self.compute_boundary(agent_h, agent_w)
        pt1 = (0.5 * ws + 0.5 * we, 0.7 * hs + 0.3 * he)
        pt2 = (0.7 * ws + 0.3 * we, 0.3 * hs + 0.7 * he)
        pt3 = (0.3 * ws + 0.7 * we, 0.3 * hs + 0.7 * he)
        draw.polygon([pt1, pt2, pt3], fill=self.agent_color)

        # draw rewards
        for h in range(self.height):
            for w in range(self.width):
                ws, we, hs, he = self.compute_boundary(h, w)
                if self.target_matrix[h, w] == 0:
                    continue
                idx = self.target_matrix[h, w] - 1
                reward_color = self.game_map.reward_colors[idx]
                draw.rectangle([(ws * 0.7 + we * 0.3, hs * 0.7 + he * 0.3),
                                (ws * 0.3 + we * 0.7, hs * 0.3 + he * 0.7)], fill=reward_color)

        image_arr = np.asarray(image)
        resized_image_arr = cv2.resize(image_arr, (self.obs_h, self.obs_w),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(resized_image_arr, dtype=np.uint8)
        return int_image


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1]

    def simulate(self, state, action):
        new_state = self._merge(state, self.delta[action])
        return new_state

    def backup(self):
        self.current_steps_backup = self.current_steps
        self.pos_backup = self.pos
        self.target_matrix_backup = copy.deepcopy(self.target_matrix)
        self.invalid_map_backup = copy.deepcopy(self.invalid_map)

    def recover(self):
        self.current_steps = self.current_steps_backup
        self.pos = self.pos_backup
        self.target_matrix = copy.deepcopy(self.target_matrix_backup)
        self.invalid_map = copy.deepcopy(self.invalid_map_backup)

    def monte_carlo_joint_return(self, obs, num_samples, initial_action, gamma=0.99,
                                 eval_policy=None, trained_agent=None):
        self.backup()
        samples = []
        for idx in range(num_samples):
            self.recover()
            time = 0
            episode_reward = np.zeros(self.reward_dim)
            while True:
                if time == 0:
                    action = initial_action
                else:
                    action = random.choice([0, 1, 2, 3])
                obs, reward, terminate, _ = self.step(action)
                episode_reward += reward * (gamma ** time)
                time += 1
                if terminate:
                    break
            samples.append(episode_reward)
        return np.stack(samples)


def play(env):
    import time
    obs = env.reset()
    env.render()
    print('action space:', env.action_space.n)
    total_r = np.zeros((env.reward_dim, ))
    discounted_r = np.zeros((env.reward_dim, ))
    t = 0

    while True:
        while True:
            try:
                a = input('action:')
                a = int(a)
                assert 0 <= a < env.action_space.n
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                import traceback
                traceback.print_exc()
        obs, r, done, _ = env.step(a)
        env.render()
        total_r += r
        discounted_r += r * (0.99 ** t)
        time.sleep(0.02)
        t += 1
        print('reward, total_reward: %s, %s' % (str(r), str(total_r)))
        if done:
            print(f'finished! total_r = {total_r}')
            print(f'discounted_r = {discounted_r}')
            obs = env.reset()
            env.render()

def show_nonblock():
    plt.ion()
    plt.show()
    plt.pause(0.001)

class GridWorldV0(GridWorld):
    def __init__(self):
        super(GridWorldV0, self).__init__(all_game_maps[0])
        
class GridWorldV1(GridWorld):
    def __init__(self):
        super(GridWorldV1, self).__init__(all_game_maps[1])
        
class GridWorldV2(GridWorld):
    def __init__(self):
        super(GridWorldV2, self).__init__(all_game_maps[2])

class GridWorldV3(GridWorld):
    def __init__(self):
        super(GridWorldV3, self).__init__(all_game_maps[3],
                                          constrain_env=True,
                                          constrain_value=[0.2, 0.7, 0.7],
                                          constrain_type=['<', '>', '>'])

class GridWorldV4(GridWorld):
    def __init__(self):
        super(GridWorldV4, self).__init__(all_game_maps[4],
                                          constrain_env=True,
                                          constrain_value=[0.2, 1.6, 0.8],
                                          constrain_type=['<', '>', '>'])

class GridWorldV5(GridWorld):
    def __init__(self):
        super(GridWorldV5, self).__init__(all_game_maps[5],
                                          constrain_env=True,
                                          constrain_value=[0.2, 0.92, 0.87],
                                          constrain_type=['<', '>', '>'])

class GridWorldV6(GridWorld):
    def __init__(self):
        super(GridWorldV6, self).__init__(all_game_maps[6],
                                          constrain_env=True,
                                          constrain_value=[0.6, 0.6, 0.6],
                                          constrain_type=['>', '>', '>'])

if __name__ == '__main__':
    # export DISPLAY=localhost:0.0

    env = GridWorldV6()
    # env = GridWorldV1()
    # env = GridWorldV2()

    play(env)
