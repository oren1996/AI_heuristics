import copy
import itertools
import logging
import math
import random

IDS = [204864532, 206202384]


# class Agent:
#     def __init__(self, initial_state, player_number):
#         self.ids = IDS
#
#     def act(self, state):
#         raise NotImplementedError


class Agent:
# class UCTAgent:
    ### 60 SECONDS FOR UCT
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.initial_state = initial_state
        self.player_number = player_number
        self.rows = len(initial_state['map'])
        self.cols = len(initial_state['map'][0])
        self.score = {'player 1': 0, 'player 2': 0}
        # self.count_to_destination = 0
        # self.number_of_passengers = len(initial_state['passengers'].keys())
        # self.not_visited = []
        self.my_taxis = []
        for taxi_name, taxi in initial_state['taxis'].items():
            if taxi['player'] == player_number:
                self.my_taxis.append(taxi_name)
        self.UCT_tree = self.build_UCT_tree(self.initial_state)
        self.root_node = self.MCTS(self.UCT_tree)
        self.game_state = copy.deepcopy(initial_state)
        self.game_node = copy.deepcopy(self.root_node)
        # self.count_depth(self.root_node)



    # def count_depth(self, root_node):
    #     if not root_node['children']:
    #         return
    #     else:
    #         self.count_depth(root_node['children'][0])
    #         print(root_node)
    #         print()
    #         return

            # for child in root_node['children']:
            #     self.count_depth(child['children'])



    def build_UCT_tree(self, initial_state):
        # for the root the action will be 'initial'
        root_node = {"state": initial_state, "parent": None, "children_actions": [], 'visit_count': 0, 'total_score': 0, 'best_action': None}
        all_actions = self.actions(initial_state)
        for action in all_actions:
            # if problem of space complexity change value of the key "parent"
            copy_parent_state = copy.deepcopy(initial_state)
            child_state = self.apply_action(copy_parent_state, action, self.player_number)
            child_node = {"state": child_state, "parent": root_node, "children_actions": [], 'visit_count': 0, 'total_score': 0,  'best_action': None}
            root_node["children_actions"].append({action: child_node})
            # self.not_visited.append(child_node)
        print(root_node)
        return root_node



    def MCTS(self, UCT_tree):
        counter = 0
        while counter < 100:
            counter += 1
            print(counter)
            current_node = UCT_tree
            # if counter == 50:
            #     print()

            while True:
                next_node = self.selection(current_node)
                # next_state = next_node['state']

                if next_node['visit_count'] == 0:
                    simulation_result = [self.simulation(next_node), next_node]
                    self.backpropagation(simulation_result)
                    break
                # else:
                #     self.expansion(UCT_tree, next_node, next_state)
                #     current_node = next_node
        return UCT_tree


    def selection(self, UCT_tree):
        # select the next node to expand using the UCT formula
        max_uct_value = -math.inf
        next_node = None
        for child_node in UCT_tree['children_actions']:
            if child_node['visit_count'] == 0:
                uct_value = math.inf
                # self.not_visited.append(child_node)
            else:
                uct_value = child_node['total_score'] / child_node['visit_count'] + math.sqrt(2 * math.log(UCT_tree['visit_count']) / child_node['visit_count'])
            if uct_value > max_uct_value:
                max_uct_value = uct_value
                next_node = child_node
        return next_node


    def expansion(self, UCT_tree, parent_node, parent_state):
        # expand the selected node by adding child nodes representing possible actions
        all_actions = self.actions(parent_state)
        for action in all_actions:
            # temp_child = {"state": copy_initial_state1, "parent": root_node, "children_actions": [], 'visit_count': 0, 'total_score': 0, 'best_action': None}
            #
            # if action not in parent_node["children"]:
            copy_parent_state = copy.deepcopy(parent_state)
            child_state = self.apply_action(copy_parent_state, action, self.player_number)
            child_node = {"state": child_state, "parent": parent_node, "children_actions": [], 'visit_count': 0, 'total_score': 0, 'best_action': None}
            parent_node["children_actions"].append(child_node)
            # self.not_visited.append(child_node)


    def simulation(self, node):
        # simulate the outcome of taking a random action and return the resulting score
        current_state = copy.deepcopy(node['state'])
        self.score = {'player 1': 0, 'player 2': 0}
        # self.count_to_destination = 0
        depth = 0
        while True:
            if self.is_terminal(current_state, depth):
                break
            else:
                all_actions = self.actions(current_state)
                random_action = random.choice(all_actions)
                next_state = self.apply_action(current_state, random_action, self.player_number)
                current_state = copy.deepcopy(next_state)
            depth += 1
        return self.score[f"player {self.player_number}"]


    def is_terminal(self, state, depth):
        # play with the stop condition
        # if self.count_to_destination == self.number_of_passengers/2:
        # if state['turns to go'] == 0:
        #     return True
        if depth == 20:
            return True
        else:
            return False


    def backpropagation(self, simulation_result):
        # update the statistics for the expanded node and its ancestors
        score = simulation_result[0]
        current_node = simulation_result[1]
        child_score = 0
        child_state = None
        while current_node is not None:
            current_node['visit_count'] += 1
            flag = False

            # we are in a leaf
            if not current_node['children']:
                current_node['total_score'] += score

            else:
                # we are not in a leaf
                # if score of a child node is greater of all the score of his brothers,
                # it means that it's the best node to choose
                for child in current_node['children']:
                    # it's the same node
                    if child['state'] == child_state:
                        continue

                    if child_score > child['total_score']:
                        flag = True

                if current_node["best_action"] is None or flag:
                    current_node["best_action"] = child_state
                    current_node['total_score'] += score

            child_score = current_node['total_score']
            child_state = current_node['action']
            current_node = current_node["parent"]





    def actions(self, state):
        all_actions = []
        for taxi_name in self.my_taxis:
            taxi_actions = []
            x, y = state['taxis'][taxi_name]['location']

            # check for move actions
            # can go up
            if x - 1 >= 0 and state['map'][x - 1][y] != 'I':
                taxi_actions.append(('move', taxi_name, (x - 1, y)))

            # can go down
            if x + 1 < self.rows and state['map'][x + 1][y] != 'I':
                taxi_actions.append(('move', taxi_name, (x + 1, y)))

            # can go left
            if y - 1 >= 0 and state['map'][x][y - 1] != 'I':
                taxi_actions.append(('move', taxi_name, (x, y - 1)))

            # can go right
            if y + 1 < self.cols and state['map'][x][y + 1] != 'I':
                taxi_actions.append(('move', taxi_name, (x, y + 1)))

            for passenger_name in self.initial_state['passengers'].keys():
                # add pick-up action
                taxi_actions.append(('pick up', taxi_name, passenger_name))

                # add drop-off action
                taxi_actions.append(('drop off', taxi_name, passenger_name))

            # add the wait action
            taxi_actions.append(('wait', taxi_name))
            all_actions.append(taxi_actions)
        all_possible_actions = list(itertools.product(*all_actions))
        all_actions = self.check_actions(state, all_possible_actions, self.player_number)
        return all_actions

    def neighbors(self, location):
        """
        return the neighbors of a location
        """
        x, y = location[0], location[1]
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for neighbor in tuple(neighbors):
            if neighbor[0] < 0 or neighbor[0] >= self.cols or neighbor[1] < 0 or neighbor[1] >= \
                    self.rows or self.initial_state['map'][neighbor[0]][neighbor[1]] != 'P':
                neighbors.remove(neighbor)
        return neighbors

    def check_actions(self, state, all_possible_actions, player):
        all_actions = []
        for possible_action in all_possible_actions:
            if self.check_if_action_legal(state, possible_action, player):
                all_actions.append(possible_action)
        return all_actions


    def check_if_action_legal(self, state, action, player):
        all_passengers = state['passengers'].keys()
        def _is_move_action_legal(move_action, player):
            taxi_name = move_action[1]
            if taxi_name not in state['taxis'].keys():
                # logging.error(f"Taxi {taxi_name} does not exist!")
                return False
            if player != state['taxis'][taxi_name]['player']:
                # logging.error(f"Taxi {taxi_name} does not belong to player {player}!")
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            if l2 not in self.neighbors(l1):
                # logging.error(f"Taxi {taxi_name} cannot move from {l1} to {l2}!")
                return False
            return True

        def _is_pick_up_action_legal(pick_up_action, player):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if player != state['taxis'][taxi_name]['player']:
                return False
            if passenger_name not in all_passengers:
                return False
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            return True

        def _is_drop_action_legal(drop_action, player):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if player != state['taxis'][taxi_name]['player']:
                return False
            if passenger_name not in all_passengers:
                return False
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['destination']:
                return False
            # check passenger is in the taxi
            if state['passengers'][passenger_name]['location'] != taxi_name:
                return False
            return True

        def _is_action_mutex(global_action):
            assert type(global_action) == tuple, "global action must be a tuple"
            # one action per taxi
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True
            # pick up the same person
            pick_actions = [a for a in global_action if a[0] == 'pick up']
            if len(pick_actions) > 1:
                passengers_to_pick = set([a[2] for a in pick_actions])
                if len(passengers_to_pick) != len(pick_actions):
                    return True
            return False

        players_taxis = self.my_taxis

        if len(action) != len(players_taxis):
            # logging.error(f"You had given {len(action)} atomic commands, while you control {len(players_taxis)}!")
            return False

        for atomic_action in action:
            # trying to act with a taxi that is not yours
            if atomic_action[1] not in players_taxis:
                logging.error(f"Taxi {atomic_action[1]} is not yours!")
                return False
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action, player):
                    # logging.error(f"Move action {atomic_action} is illegal!")
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action, player):
                    # logging.error(f"Pick action {atomic_action} is illegal!")
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action, player):
                    # logging.error(f"Drop action {atomic_action} is illegal!")
                    return False
            # elif atomic_action[0] != 'wait':
            #     return False
            elif atomic_action[0] == 'wait':
                if len(all_passengers) > 0:
                    return False
            else:
                return False
        # check mutex action
        if _is_action_mutex(action):
            # logging.error(f"Actions {action} are mutex!")
            return False


        # -------------------------------------------------------------------------

        # # Not sure if I should check for collision between MY taxis or between every taxis in the game
        # if len(self.initial_state['taxis']) > 1:
        #     taxis_location_dict = dict(
        #         [(t, self.initial_state['taxis'][t]['location']) for t in self.initial_state['taxis'].keys()])
        #     move_actions = [a for a in action if a[0] == 'move']
        #     for move_action in move_actions:
        #         taxis_location_dict[move_action[1]] = move_action[2]
        #     if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
        #         # logging.error(f"Actions {action} cause collision!")
        #         return False

        # -------------------------------------------------------------------------

        # check taxis collision
        if len(self.my_taxis) > 1:
            taxis_location_dict = dict([(t, self.initial_state['taxis'][t]['location']) for t in self.my_taxis])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                # logging.error(f"Actions {action} cause collision!")
                return False
        return True


    def apply_action(self, state, action, player):
        for atomic_action in action:
            self._apply_atomic_action(state, atomic_action, player)
        state['turns to go'] -= 1
        return state

    def _apply_atomic_action(self, state, atomic_action, player):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state['taxis'][taxi_name]['location'] = atomic_action[2]
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            state['taxis'][taxi_name]['capacity'] -= 1
            state['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            state['taxis'][taxi_name]['capacity'] += 1
            self.score[f"player {player}"] += self.initial_state['passengers'][passenger_name]['reward']
            # self.count_to_destination += 1

            # DO I NEED TO DELETE THE PASSENGER FROM THE INITIAL STATE AS WELL ???
            del state['passengers'][passenger_name]
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    ### 5 SECONDS
    def act(self, state):
        # if state == self.game_state:
        best_action = self.game_node['best_action']
        children = self.game_node['children']
        for child in children:
            if child['action'] == best_action:
                self.game_node = children[children.index(child)]
                break
        # print(state)
        # print(best_action)
        # self.game_state = self.apply_action(state, best_action, self.player_number)
        return best_action



