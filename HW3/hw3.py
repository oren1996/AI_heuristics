import copy
import itertools
import math
import random
import networkx as nx

IDS = [204864532, 206202384]


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.agent = UCTAgent(initial_state, player_number)

    def act(self, state):
        return self.agent.act(state)


# class Agent:
class UCTAgent:
    ### 60 SECONDS FOR UCT
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.initial_state = initial_state
        self.initial_state_improved = copy.deepcopy(initial_state)
        self.my_taxis = []
        for taxi_name in initial_state['taxis'].keys():
            self.initial_state_improved['taxis'][taxi_name]['max_capacity'] = self.initial_state['taxis'][taxi_name]['capacity']
            if initial_state['taxis'][taxi_name]['player'] == player_number:
                self.my_taxis.append(taxi_name)
            else:
                del self.initial_state_improved['taxis'][taxi_name]
        self.player_number = player_number
        self.rows = len(initial_state['map'])
        self.cols = len(initial_state['map'][0])
        self.score = {'player 1': 0, 'player 2': 0}
        self.distances = Distances(initial_state)
        self.UCT_tree = self.build_UCT_tree(self.initial_state)
        self.root_node = self.MCTS(self.UCT_tree)
        self.game_state = copy.deepcopy(initial_state)
        self.game_node = copy.deepcopy(self.root_node)


    def build_UCT_tree(self, initial_state):
        taxi_actions = []
        for taxi_name in self.my_taxis:
            taxi_actions.append(('initial', taxi_name, initial_state['taxis'][taxi_name]['location']))
        initial = tuple(taxi_actions)
        # for the root the action will be 'initial'
        root_node = {"action": initial, "parent": None, "children": [], 'visit_count': 0, 'total_score': 0, 'best_action': None}
        all_actions = self.actions(initial_state)
        for action in all_actions:
            # if problem of space complexity change value of the key "parent"
            child_node = {"action": action, "parent": root_node, "children": [], 'visit_count': 0, 'total_score': 0,  'best_action': None}
            root_node["children"].append(child_node)
        return root_node

    def MCTS(self, UCT_tree):
        counter = 0
        while counter < 27500:
            counter += 1
            if counter % 5000 == 0:
                print(counter)
            current_node = UCT_tree
            copy_state = copy.deepcopy(self.initial_state)
            while True:
                next_node = self.selection(UCT_tree, current_node)
                next_state = self.apply_action(copy_state, next_node['action'], self.player_number)

                if next_node['visit_count'] == 0:
                    simulate = self.check_expand_simulate(next_node)
                    if simulate:
                        simulation_result = [self.simulation(next_state), next_node]
                        self.backpropagation(simulation_result)
                        break

                else:
                    if len(next_node['children']) == 0:
                        expand = self.check_expand_simulate(next_node)
                        if expand:
                            self.expansion(UCT_tree, next_node, next_state)
                    current_node = next_node
                    copy_state = next_state

        return UCT_tree


    def selection(self, UCT_tree, current_node):
        # select the next node to expand using the UCT formula
        max_uct_value = -math.inf
        next_node = None
        C = 9.8
        for child_node in current_node['children']:
            if child_node['visit_count'] == 0:
                uct_value = math.inf
            else:
                uct_value = C * (child_node['total_score'] / child_node['visit_count']) + math.sqrt(2 * math.log(UCT_tree['visit_count']) / child_node['visit_count'])
            if uct_value > max_uct_value:
                max_uct_value = uct_value
                next_node = child_node
        return next_node


    def expansion(self, UCT_tree, parent_node, parent_state):
        # expand the selected node by adding child nodes representing possible actions
        all_actions = self.actions(parent_state)
        for action in all_actions:
            child_node = {"action": action, "parent": parent_node, "children": [], 'visit_count': 0, 'total_score': 0, 'best_action': None}
            parent_node["children"].append(child_node)


    def simulation(self, state):
        # simulate the outcome of taking a random action and return the resulting score
        current_state = copy.deepcopy(state)
        self.score = {'player 1': 0, 'player 2': 0}
        counter = 0
        while True:
            if self.is_terminal(current_state, counter):
                break
            else:
                all_actions = self.build_actions(current_state)
                next_action = self.choose_action(current_state, all_actions)
                next_state = self.apply_action(current_state, next_action, self.player_number)
                current_state = copy.deepcopy(next_state)
            counter += 1
        return self.score[f"player {self.player_number}"]


    def backpropagation(self, simulation_result):
        # update the statistics for the expanded node and its ancestors
        score = simulation_result[0]
        current_node = simulation_result[1]
        child_score = 0
        child_action = None
        while current_node is not None:
            current_node['visit_count'] += 1
            flag = True

            # we are not in a leaf
            if current_node['children']:
                # we are not in a leaf
                # if score of a child node is greater of all the score of his brothers,
                # it means that it's the best node to choose
                for child in current_node['children']:
                    # it's the same node
                    if child['action'] == child_action:
                        continue

                    if child_score <= child['total_score']:
                        flag = False
                        break

                if current_node["best_action"] is None or flag:
                    current_node["best_action"] = child_action

            current_node['total_score'] += score
            child_score = current_node['total_score']
            child_action = current_node['action']
            current_node = current_node["parent"]

    def build_mini_UCT_tree(self, initial_state):
        taxi_actions = []
        for taxi_name in self.my_taxis:
            taxi_actions.append(('initial', taxi_name, initial_state['taxis'][taxi_name]['location']))
        initial = tuple(taxi_actions)

        # for the root the action will be 'initial'
        root_node = {"action": initial, "parent": None, "children": [], 'visit_count': 0, 'total_score': 0,
                     'best_action': None}
        all_actions = self.build_actions(initial_state)
        root_node['children'] = all_actions
        return root_node

    def check_expand_simulate(self, current_node):
        if current_node['parent']['parent'] is None:
            counter_action = 0
            for taxi_action in current_node['action']:
                if taxi_action[0] == 'move':
                    if taxi_action[2] == current_node['parent']['action'][counter_action][2]:
                        return False
                    counter_action += 1
            return True
        else:
            counter_action = 0
            for taxi_action in current_node['action']:
                if taxi_action == current_node['parent']['action'][counter_action]:
                    return False
                counter_action += 1
            return True

    def is_terminal(self, state, counter):
        if counter == 5:
            return True
        else:
            return False

    def build_actions(self, state):
        actions = {}
        for taxi in self.my_taxis:
            actions[taxi] = set()
            neighboring_tiles = self.neighbors(state["taxis"][taxi]["location"])

            for tile in neighboring_tiles:
                actions[taxi].add(("move", taxi, tile))
            if state["taxis"][taxi]["capacity"] > 0:
                for passenger in state["passengers"].keys():
                    if state["passengers"][passenger]["location"] == state["taxis"][taxi]["location"]:
                        actions[taxi].add(("pick up", taxi, passenger))
            for passenger in state["passengers"].keys():
                if (state["passengers"][passenger]["destination"] == state["taxis"][taxi]["location"]
                        and state["passengers"][passenger]["location"] == taxi):
                    actions[taxi].add(("drop off", taxi, passenger))
            if len(state['passengers'].keys()) <= 1:
                actions[taxi].add(("wait", taxi))
        return actions

    def choose_action(self, state, actions):
        flag = False
        counter = 0
        while True:
            if counter == 10:
                taxi_actions = []
                for taxi_name in self.my_taxis:
                    taxi_actions.append(('wait', taxi_name))
                taxi_actions = tuple(taxi_actions)
                return taxi_actions
            whole_action = []
            passenger_already_chosen = []
            for atomic_actions in actions.values():
                for action in atomic_actions:
                    if action[0] == "drop off":
                        whole_action.append(action)
                        break
                    if action[0] == "pick up":
                        whole_action.append(action)
                        passenger_already_chosen.append(action[2])
                        break
                else:
                    if flag:
                        whole_action.append(random.choice(list(atomic_actions)))
                    else:
                        whole_action.append(self.best_action_taxi_i(state, atomic_actions, passenger_already_chosen))
            whole_action = tuple(whole_action)
            if self.check_if_action_legal(state, whole_action, self.player_number):
                return whole_action
            else:
                flag = True
            counter += 1

    def best_action_taxi_i(self, current_state, all_actions_taxi_i, passenger_already_chosen):
        diameter = self.distances.diameter
        best_action = None
        best_reward = -math.inf
        second_best_action = None
        second_best_reward = -math.inf
        close_passenger_name = None
        second_close_passenger_name = None
        closest_passenger_name = None
        for taxi_action in all_actions_taxi_i:
            reward = 0
            act = taxi_action[0]
            taxi_name = taxi_action[1]
            if act == 'move':
                future_taxi_location = taxi_action[2]
                # move to pick up
                if current_state['taxis'][taxi_name]['capacity'] == self.initial_state_improved['taxis'][taxi_name][
                    'max_capacity']:
                    closest_passenger_location, closest_passenger_name = self.find_closest_passenger(current_state,
                                                                                                     taxi_name,
                                                                                                     passenger_already_chosen)
                    reward = diameter - self.distances.check_distances(self.distances.shortest_path_distances,
                                                                       future_taxi_location, closest_passenger_location)
                # move to drop off
                else:
                    passenger_destination_location = self.passenger_destination(current_state, taxi_name)
                    reward = diameter - self.distances.check_distances(self.distances.shortest_path_distances,
                                                                       future_taxi_location,
                                                                       passenger_destination_location)
            elif act == 'wait':
                if len(current_state['passengers'].keys()) > 0:
                    reward = -diameter
                else:
                    reward = diameter

            if reward == best_reward:
                second_best_reward = reward
                second_best_action = taxi_action
                if close_passenger_name != closest_passenger_name:
                    second_close_passenger_name = closest_passenger_name

            if reward > best_reward:
                best_reward = reward
                best_action = taxi_action
                close_passenger_name = closest_passenger_name

        if best_reward == second_best_reward:
            best_action = random.choice((best_action, second_best_action))

        passenger_already_chosen.append(close_passenger_name)
        return best_action

    def find_closest_passenger(self, current_state, taxi_name, passenger_already_chosen):
        closest_passenger_location = None
        closest_passenger_name = None
        closest_distance = math.inf
        for passenger_name in current_state['passengers'].keys():
            distance_passenger = self.distances.check_distances(self.distances.shortest_path_distances,
                                                                current_state['taxis'][taxi_name]['location'],
                                                                current_state['passengers'][passenger_name]['location'])
            if distance_passenger < closest_distance and passenger_name not in passenger_already_chosen:
                closest_passenger_location = current_state['passengers'][passenger_name]['location']
                closest_distance = distance_passenger
                closest_passenger_name = passenger_name
        return closest_passenger_location, closest_passenger_name

    def passenger_destination(self, current_state, taxi_name):
        passenger_destination_location = None
        for passenger_name in current_state['passengers'].keys():
            if current_state['passengers'][passenger_name]['location'] == taxi_name:
                passenger_destination_location = current_state['passengers'][passenger_name]['destination']
                break
        return passenger_destination_location


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

    def check_if_action_legal(self,state, action, player):
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

        players_taxis = [taxi for taxi in state['taxis'].keys() if state['taxis'][taxi]['player'] == player]

        if len(action) != len(players_taxis):
            # logging.error(f"You had given {len(action)} atomic commands, while you control {len(players_taxis)}!")
            return False
        for atomic_action in action:
            # trying to act with a taxi that is not yours
            if atomic_action[1] not in players_taxis:
                # logging.error(f"Taxi {atomic_action[1]} is not yours!")
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
            elif atomic_action[0] == 'wait':
                if len(all_passengers) > 0:
                    return False
            else:
                return False
        # check mutex action
        if _is_action_mutex(action):
            # logging.error(f"Actions {action} are mutex!")
            return False
        # check taxis collision
        if len(state['taxis']) > 1:
            taxis_location_dict = dict(
                [(t, state['taxis'][t]['location']) for t in state['taxis'].keys()])
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
            del state['passengers'][passenger_name]
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented


    ### 5 SECONDS
    def act(self, state):
        best_action = self.game_node['best_action']
        # no more nodes so we need to simulate
        if best_action is None:
            mini_UCT_tree = self.build_mini_UCT_tree(state)
            best_action = self.choose_action(state, mini_UCT_tree['children'])
            return best_action
        else:
            children = self.game_node['children']
            for child in children:
                if child['action'] == best_action:
                    self.game_node = children[children.index(child)]
                    break
            return best_action


class Distances:
    def __init__(self, initial):
        self.gaStation = None
        self.state = initial
        self.graph = self.build_graph(initial)
        self.shortest_path_distances = self.create_shortest_path_distances(self.graph)
        self.diameter = nx.diameter(self.graph)

    def build_graph(self, initial):
        """
        build the graph of the problem
        """
        n, m = len(initial['map']), len(initial['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if initial['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def create_shortest_path_distances(self, graph):
        d = {}
        for n1 in graph.nodes:
            for n2 in graph.nodes:
                d[(n1, n2)] = len(nx.shortest_path(graph, n1, n2)) - 1
        return d


    def check_distances(self, graph, node1, node2):
        if (node1, node2) not in graph:
            return self.diameter
        else:
            return graph[(node1, node2)]
