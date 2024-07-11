import numpy as np
from .sokoban_env import SokobanEnv, CHANGE_COORDINATES
from gym.spaces import Box
from gym.spaces.discrete import Discrete
import copy
import hashlib


class PushAndPullSokobanEnv(SokobanEnv):

    def __init__(self,
             dim_room=(10, 10),
             max_steps=120,
             num_boxes=3,
             num_gen_steps=None,
             regen_room = False,
             observation = 'rgb_array'):

        super(PushAndPullSokobanEnv, self).__init__(dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes, num_gen_steps=num_gen_steps, reset=False, regen_room=regen_room, observation=observation)
        screen_height, screen_width = (dim_room[0], dim_room[1])
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.boxes_are_on_target = [False] * num_boxes
        self.action_space = Discrete(len(ACTION_LOOKUP))

        _ = self.reset(self.regen_room)

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        prev_dist = self._calc_box_distance_from_target()
        prev_player_close_to_box = None
        if self.num_boxes == 1:
            prev_player_close_to_box = self._calc_box_distance_from_player()
        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        # All push actions are in the range of [0, 3]
        if action < 4:
            moved_player, moved_box = self._push(action)
        else:
            moved_player, moved_box = self._pull(action)

        self._calc_reward()

        # Getting player to box proximity
        if self.num_boxes == 1:
            self._player_proximity_reward_calc(prev_player_close_to_box)
        # Getting closer reward
        self._box_getting_closer_reward_calc(prev_dist)

        done = self._check_if_done()

        # Convert the observation to our observation (RGB) frame
        observation = self.render(mode= self.observation)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
            self.add_result(self._check_if_all_boxes_on_target())

        # Rewarding great behaviour -> less steps finish = more points
        if self._check_if_all_boxes_on_target():
            self.reward_last += self.reward_less_steps() * self.reward_finished
            self.games_won = self.games_won + 1 #JUST FOR PRINTING

        return observation, self.reward_last, done, info
    
    # Just for debugging#######
    def percentage_won(self):
        return self.games_won / self.games_played
    ########################

    def past_games_percentage_won(self):
        return self.past_games.count(True) / len(self.past_games)

    def add_result(self,new_result):
        self.past_games.append(new_result)
        if len(self.past_games) > 50:
            self.past_games.pop(0)  # Remove the oldest result

    def _reward_player_close_to_box(self):
        if self._calc_box_distance_from_target() == 1:
            self.reward_last += self.player_close_to_box_reward
        else:
            self.reward_last += self.player_far_from_box_reward
        
    def _calc_current_observation_reward(self, observation):        
        obs_hash = self.hash_observation(observation)
        if obs_hash not in self.obs_dict:
            self.obs_dict[obs_hash] = observation
            self.reward_last += self.existing_observation_reward
       
    def reward_less_steps(self):
        return 2 - (self.num_env_steps / 500)

    def _box_getting_closer_reward_calc(self, prev_dist):
        after_dist = self._calc_box_distance_from_target()
        if after_dist > -1 and prev_dist > -1:
            if after_dist < prev_dist:
                self.reward_last += self.reward_less_steps() * self.box_getting_closer_to_target_reward
            elif after_dist > prev_dist:
                self.reward_last += self.reward_less_steps() * self.box_getting_farther_from_target_reward
                
    def _player_proximity_reward_calc(self, prev_player_close_to_box):
        after_player_close_to_box = self._calc_box_distance_from_player()
        if after_player_close_to_box > -1 and prev_player_close_to_box > -1:
            if after_player_close_to_box < prev_player_close_to_box:         
                self.reward_last += self.reward_less_steps() * self.player_getting_closer_to_box_reward
            elif after_player_close_to_box > prev_player_close_to_box:
                self.reward_last += self.reward_less_steps() * self.player_getting_farther_from_box_reward

    def _calc_box_distance_from_player(self):
        box_location = self._find_box_location()[0]
        if box_location is None or self.player_position is None:
            return -1

        distance = max((box_location[0] - self.player_position[0]),(box_location[1] - self.player_position[1]))
        return distance
    
    def _calc_box_distance_from_target(self):
        box_location = self._find_box_location()
        target_location = self._find_target_location()
        if box_location is None or target_location is None:
            return -1

        distance = np.sum((box_location - target_location)**2) #no need to square root
        return distance
    
    def _find_target_location(self):
        idx = np.argwhere(self.room_state == 2)
        if len(idx) > 0:
            self.current_target_pos = np.asarray([np.asarray([loc[0], loc[1]]) for loc in idx])
        return self.current_target_pos

    def _find_box_location(self):
        idx = np.argwhere(self.room_state == 4)
        if len(idx) > 0:
            self.current_box_pos = np.asarray([np.asarray([loc[0], loc[1]]) for loc in idx])
        return self.current_box_pos

    def _pull(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if box_next_to_player:
                # Move Box
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]

            return True, box_next_to_player

        return False, False

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    # 0: 'no operation',
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    # 5: 'move up',
    # 6: 'move down',
    # 7: 'move left',
    # 8: 'move right',
    4: 'pull up',
    5: 'pull down',
    6: 'pull left',
    7: 'pull right',
}

