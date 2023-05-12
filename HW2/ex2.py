import copy
import itertools
import json
import random
import networkx as nx


ids = ["204864532", "206202384"]

# initial: { "optimal": True,
#           "map": [['P', 'P', 'P'],
#                   ['P', 'G', 'P'],
#                   ['P', 'P', 'P']],
#           "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1}},
#           "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
#                   "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}},
#           "turns to go": 100 }


# initial_improved: { "optimal": True,
#           "map": [['P', 'P', 'P'],
#                   ['P', 'G', 'P'],
#                   ['P', 'P', 'P']],
#           "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1, 'max_fuel': 10, 'max_capacity": 1}},
#           "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
#                   "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}},
#           "turns to go": 100
#           "number_taxis" : 1
#           "number_passengers" : 1}



class OptimalTaxiAgent:

    def __init__(self, initial):
        self.initial = initial
        self.initial_improved = copy.deepcopy(initial)
        self.initial_improved['number_taxis'] = len(initial['taxis'].keys())
        self.initial_improved['number_passengers'] = len(initial['passengers'].keys())
        for taxi in initial['taxis'].keys():
            self.initial_improved['taxis'][taxi]['max_fuel'] = initial['taxis'][taxi]['fuel']
            self.initial_improved['taxis'][taxi]['max_capacity'] = initial['taxis'][taxi]['capacity']
        self.distances = Distances(initial)
        self.possible_destination = None
        self.best_actions = {}
        self.value_iteration()
    def act(self, state):
        new_state = {key: state[key] for key in self.order}
        action = self.best_actions[json.dumps(new_state)]
        return action


    def value_iteration(self):
        states = self.value_state_initialization()
        values_t_new = self.initialization_initial_values(states)
        all_possible_actions = self.create_randoms_action()
        turns_to_go = self.initial['turns to go']

        for i in range(0, turns_to_go):
            print(f"iteration: {i}")
            values_t_past = copy.deepcopy(values_t_new)
            values_t_new = dict()
            for state in states[i]:

                # possible actions for the state
                actions = self.actions(state, all_possible_actions)

                # for each action compute all states prime we can go to, given the current state
                actions_states = dict()
                for action in actions:
                    actions_states[action] = self.compute_states_prime(state, action)
                    if action == 'terminate':
                        actions_states[action] = []

                # compute the best action for the given state
                best_action = self.compute_best_action(state, actions_states, values_t_past)

                # save the states_prime we can go to by applying best_action
                states_prime = actions_states[best_action]

                # compute transition probabilities to move to all states_prime by applying best_action
                # how it looks: {state_prime:prob, ...}
                transition_probs = self.compute_transition_probs(state, best_action, states_prime)

                # compute the sigma
                sigma = self.compute_sigma(states_prime, transition_probs, values_t_past)

                reward_state = self.reward(best_action)

                key_state = json.dumps(state)
                new_value = reward_state + sigma
                temp = {key_state: new_value}
                temp2 = {key_state: best_action}
                values_t_new.update(temp)
                self.best_actions.update(temp2)

    def initialization_initial_values(self, states):
        initial_values = {}
        for state in states[0]:
            initial_values[json.dumps(state)] = 0
        return initial_values

    def compute_states_prime(self, state, action):
        passengers = state['passengers'].keys()
        possible_destinations = self.possible_destination
        number_of_possible_states = len(possible_destinations)
        copy_state = copy.deepcopy(state)
        self.result(copy_state, action)
        s_primes = []
        for i in range(number_of_possible_states):
            counter_passenger = 0
            copy_state_prime = copy.deepcopy(copy_state)
            for passenger_name in passengers:
                copy_state_prime['passengers'][passenger_name]['destination'] = possible_destinations[i][counter_passenger]
                counter_passenger += 1
                if copy_state_prime not in s_primes:
                    s_primes.append(copy_state_prime)

        if action == 'reset':
            for s in s_primes:
                flag = False
                for passenger_name in passengers:
                    if s['passengers'][passenger_name]['destination'] != self.initial['passengers'][passenger_name]['destination']:
                        flag = True
                if flag:
                    s_primes.remove(s)
        return s_primes

    def compute_optimal_action(self, state, values_t_new):

        # compute all actions given the state
        all_possible_actions = self.create_randoms_action()
        actions = self.actions(state, all_possible_actions)

        # for each action compute all states prime we can go to, given the current state
        actions_states = dict()
        for action in actions:
            actions_states[action] = self.compute_states_prime(state, action)
            if action == 'terminate':
                actions_states[action] = []

        # compute the best action for the given state
        best_action = self.compute_best_action(state, actions_states, values_t_new)

        return best_action

    def reward(self, action):
        if action[0][0] == 'pick up':
            return 0
        elif action[0][0] == 'drop off':
            return 100
        elif action[0][0] == 'refuel':
            return -10
        elif action == 'reset':
            return -50
        elif action == 'terminate':
            return 0
        elif action[0][0] == 'wait':
            return 0
        elif action[0][0] == 'move':
            return 0

    def compute_best_action(self, state, actions_states, values_t):
        # first, for each action compute its value
        action_values = {}
        actions = actions_states.keys()
        for action in actions:
            s_primes = actions_states[action]
            action_value = self.compute_action_value(state, action, s_primes, values_t) + self.reward(action)
            action_values[action] = action_value

        # find the best action with the greatest action_value
        best_action = max(action_values, key=action_values.get)
        return best_action

    def compute_action_value(self, state, action, s_primes, values_t):
        probs = self.compute_transition_probs(state, action, s_primes)
        v = 0

        for s in s_primes:
            transition_prob = probs[json.dumps(s)]
            if s['turns to go'] == 0:
                v_past = 0
            else:
                v_past = values_t[json.dumps(s)]
            v += transition_prob * v_past

        return v

    def compute_sigma(self, states_prime, transition_probs, values_t_past):
        v = 0
        for s in states_prime:
            transition_prob = transition_probs[json.dumps(s)]
            if s['turns to go'] == 0:
                v_past = 0
            else:
                v_past = values_t_past[json.dumps(s)]
            v += transition_prob * v_past

        return v

    def compute_transition_probs(self, state, action, states_prime):
        # need to be compatability between states_prime and the probs returned.
        # returns dict - prob for each state_prime
        probs = {}
        passengers = state['passengers'].keys()
        for passenger_name in passengers:
            n = len(self.initial['passengers'][passenger_name]['possible_goals'])
            prob_change_goal = state['passengers'][passenger_name]['prob_change_goal']
            prob_transition = prob_change_goal / n
            if state['passengers'][passenger_name]['destination'] in state['passengers'][passenger_name]['possible_goals']:
                for s_prime in states_prime:
                    state_destination = state['passengers'][passenger_name]['destination']
                    s_prime_destination = s_prime['passengers'][passenger_name]['destination']
                    if s_prime_destination == state_destination:
                        probs[json.dumps(s_prime)] = 1 - prob_change_goal + prob_transition
                    else:
                        probs[json.dumps(s_prime)] = prob_transition
                    if action == 'reset':
                        probs[json.dumps(s_prime)] = 1
            else:
                for s_prime in states_prime:
                    state_destination = state['passengers'][passenger_name]['destination']
                    s_prime_destination = s_prime['passengers'][passenger_name]['destination']
                    if s_prime_destination == state_destination:
                        probs[json.dumps(s_prime)] = 1 - prob_change_goal
                    else:
                        probs[json.dumps(s_prime)] = prob_transition
                    if action == 'reset':
                        probs[json.dumps(s_prime)] = 1
        return probs

    def value_state_initialization(self):
        state1 = self.initial_improved
        states = list()
        self.n = len(self.initial_improved['map'])
        self.m = len(self.initial_improved['map'][0])
        turns_to_go = list(range(1,self.initial['turns to go'] + 1))
        taxis_names = [taxi_name for taxi_name in state1['taxis'].keys()]
        number_taxis = state1['number_taxis']
        fuel_taxis = []
        capacity_taxis = []
        possible_location_taxis = number_taxis * [[(i, j) for i in range(self.n) for j in range(self.m) if self.initial_improved['map'][i][j] != 'I']]

        possible_location_passengers = []
        possible_destination_passengers = []
        for passenger_name in state1['passengers'].keys():
            poss_location = list()
            poss_destination = list()
            for possible_destination in state1['passengers'][passenger_name]['possible_goals']:
                poss_location.append(possible_destination)
                poss_destination.append(possible_destination)
            poss_location += taxis_names
            if state1['passengers'][passenger_name]['location'] not in poss_location:
                poss_location.append(state1['passengers'][passenger_name]['location'])
            if state1['passengers'][passenger_name]['destination'] not in poss_destination:
                poss_location.append(state1['passengers'][passenger_name]['destination'])
                poss_destination.append(state1['passengers'][passenger_name]['destination'])
            possible_location_passengers.append(poss_location)
            possible_destination_passengers.append(poss_destination)

        for taxi_name in state1['taxis'].keys():
            fuel_taxis += [[i for i in range(state1['taxis'][taxi_name]['max_fuel'] + 1)]]
            capacity_taxis += [[i for i in range(state1['taxis'][taxi_name]['max_capacity'] + 1)]]

        all_possible_location_taxis = list(itertools.product(*possible_location_taxis))
        all_possible_location_passengers = list(itertools.product(*possible_location_passengers))
        all_possible_destination_passengers = list(itertools.product(*possible_destination_passengers))
        all_fuels = list(itertools.product(*fuel_taxis))
        all_capacity = list(itertools.product(*capacity_taxis))


        if number_taxis > 1:
            for taxis_locations in all_possible_location_taxis:
                taxi_loc = taxis_locations[0]
                for i in range(1, len(taxis_locations)):
                    if taxis_locations[i] == taxi_loc:
                        all_possible_location_taxis.remove(taxis_locations)

        for turn in turns_to_go:
            turn_to_go_i = []
            for location_taxis in all_possible_location_taxis:
                for location_passengers in all_possible_location_passengers:
                    for destination_passengers in all_possible_destination_passengers:
                        for taxis_fuel in all_fuels:
                            for taxis_capacity in all_capacity:
                                # Fill the state
                                taxi_counter = 0
                                passenger_counter = 0
                                state_i = dict()
                                state_i['optimal'] = state1['optimal']
                                state_i['map'] = state1['map']
                                state_i['taxis'] = {}
                                state_i['passengers'] = {}
                                # state_i['turns to go'] = self.initial_improved['turns to go']
                                for taxi_name in state1['taxis'].keys():
                                    state_i['taxis'][taxi_name] = {}
                                    state_i['taxis'][taxi_name]['names_passengers_aboard'] = []
                                    state_i['taxis'][taxi_name]["location"] = location_taxis[taxi_counter]
                                    state_i['taxis'][taxi_name]["fuel"] = taxis_fuel[taxi_counter]
                                    state_i['taxis'][taxi_name]["capacity"] = taxis_capacity[taxi_counter]

                                    taxi_counter += 1

                                for passenger_name in state1['passengers'].keys():
                                    state_i['passengers'][passenger_name] = {}
                                    state_i['passengers'][passenger_name]["location"] = location_passengers[passenger_counter]
                                    state_i['passengers'][passenger_name]["destination"] = destination_passengers[passenger_counter]
                                    state_i['passengers'][passenger_name]["possible_goals"] = state1['passengers'][passenger_name]["possible_goals"]
                                    state_i['passengers'][passenger_name]["prob_change_goal"] = state1['passengers'][passenger_name]["prob_change_goal"]


                                    if state_i['passengers'][passenger_name]['location'] in taxis_names:
                                        taxi_name_idx = taxis_names.index(state_i['passengers'][passenger_name]['location'])
                                        state_i['taxis'][taxis_names[taxi_name_idx]]['names_passengers_aboard'].append(passenger_name)

                                    passenger_counter += 1
                                state_i['turns to go'] = turn
                                flag_1 = False
                                flag_2 = False
                                flag_3 = False
                                for taxi_name in state_i['taxis'].keys():
                                    if len(state_i['taxis'][taxi_name]['names_passengers_aboard']) > \
                                            state1['taxis'][taxi_name]['max_capacity']:
                                        flag_1 = True
                                    if len(state_i['taxis'][taxi_name]['names_passengers_aboard']) == 0 and \
                                            state_i['taxis'][taxi_name]['capacity'] == 0:
                                        flag_2 = True
                                    if len(state_i['taxis'][taxi_name]['names_passengers_aboard']) != 0 and \
                                            state_i['taxis'][taxi_name]['capacity'] == state1['taxis'][taxi_name][
                                        'max_capacity']:
                                        flag_3 = True
                                    del state_i['taxis'][taxi_name]['names_passengers_aboard']

                                if not flag_1 and not flag_2 and not flag_3:
                                    turn_to_go_i.append(state_i)

            states.append(turn_to_go_i)

        self.possible_destination = self.possible_goals()
        self.order = list(states[0][0].keys())

        return states

    def possible_goals(self):
        destination_goals_passengers = []
        for passenger_name in self.initial_improved['passengers'].keys():
            destination_goal_passenger = [self.initial_improved['passengers'][passenger_name]['possible_goals']]
            destination_goals_passengers += destination_goal_passenger
        return list(itertools.product(*destination_goals_passengers))
    def create_randoms_action(self):
        m = len(self.initial_improved['map'])
        n = len(self.initial_improved['map'][0])
        actions = []

        for taxi_name in self.initial['taxis'].keys():
            taxis_actions = []
            for i in range(m):
                for j in range(n):
                    taxis_actions.append(('move', taxi_name, (i, j)))
            for passenger_name in self.initial['passengers'].keys():
                taxis_actions.append(('pick up', taxi_name, passenger_name))
                taxis_actions.append(('drop off', taxi_name, passenger_name))
            taxis_actions.append(('refuel', taxi_name))
            taxis_actions.append(('wait', taxi_name))
            actions.append(taxis_actions)

        all_possible_actions = list(itertools.product(*actions))
        all_possible_actions.append('reset')
        all_possible_actions.append('terminate')
        return all_possible_actions

    def actions(self, state1, all_possible_actions):
        all_actions = []
        for possible_action in all_possible_actions:
            if self.is_action_legal(state1, possible_action):
                all_actions.append(possible_action)
        return all_actions

    def is_action_legal(self, state, action):
        """
        check if the action is legal
        """
        def _is_move_action_legal(move_action):
            taxi_name = move_action[1]
            if taxi_name not in state['taxis'].keys():
                return False
            if state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            return l2 in list(self.distances.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            # check passenger is not in his destination
            if state['passengers'][passenger_name]['destination'] == state['passengers'][passenger_name]['location']:
                return False
            # check passenger not on taxi already
            if state['passengers'][passenger_name]['location'] == taxi_name:
                return False
            return True

        def _is_drop_action_legal(drop_action):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if state['taxis'][taxi_name]['location'] == state['passengers'][passenger_name]['destination'] and \
                    state['passengers'][passenger_name]['location'] == taxi_name and \
                    state['taxis'][taxi_name]['capacity'] < self.initial_improved['taxis'][taxi_name]['max_capacity']:
                return True
            return False


        def _is_refuel_action_legal(refuel_action):
            """
            check if taxi in gas location
            """
            taxi_name = refuel_action[1]
            i, j = state['taxis'][taxi_name]['location']
            if self.initial['map'][i][j] == 'G':
                return True
            else:
                return False

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

        if action == "reset":
            return True
        if action == "terminate":
            return True
        if len(action) != len(state["taxis"].keys()):
            return False
        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action):
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action):
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action):
                    return False
            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action):
                    return False
            elif atomic_action[0] != 'wait':
                return False
        # check mutex action
        if _is_action_mutex(action):
            return False
        # check taxis collision
        if len(state['taxis']) > 1:
            taxis_location_dict = dict([(t, state['taxis'][t]['location']) for t in state['taxis'].keys()])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                return False
        return True

    def result(self, state, action):
        """"
        update the state according to the action
        """
        self.apply(state, action)
        if action != "reset":
            self.environment_step(state)

    def apply(self, state, action):
        """
        apply the action to the state
        """
        if action == "reset":
            self.reset_environment(state)
            return
        if action == "terminate":
            self.terminate_execution()
            return
        for atomic_action in action:
            self.apply_atomic_action(state, atomic_action)

    def apply_atomic_action(self, state, atomic_action):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state['taxis'][taxi_name]['location'] = atomic_action[2]
            state['taxis'][taxi_name]['fuel'] -= 1
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            state['taxis'][taxi_name]['capacity'] -= 1
            state['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            if state['taxis'][taxi_name]['capacity'] < self.initial_improved['taxis'][taxi_name]['max_capacity']:
                passenger_name = atomic_action[2]
                state['passengers'][passenger_name]['location'] = state['taxis'][taxi_name]['location']
                state['taxis'][taxi_name]['capacity'] += 1
            return
        elif atomic_action[0] == 'refuel':
            state['taxis'][taxi_name]['fuel'] = self.initial_improved['taxis'][taxi_name]['fuel']
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def environment_step(self, state):
        """
        update the state of environment randomly
        """
        for p in state['passengers']:
            passenger_stats = state['passengers'][p]
            if random.random() < passenger_stats['prob_change_goal']:
                # change destination
                passenger_stats['destination'] = random.choice(passenger_stats['possible_goals'])
        state["turns to go"] -= 1
        return

    def reset_environment(self, state):
        """
        reset the state of the environment
        """
        state["taxis"] = copy.deepcopy(self.initial["taxis"])
        state["passengers"] = copy.deepcopy(self.initial["passengers"])
        state["turns to go"] -= 1
        return

    def terminate_execution(self):
        """
        terminate the execution of the problem
        """
        return

class TaxiAgent:
    def __init__(self, initial):
        self.possible_destination = None
        self.initial = initial
        self.distances = Distances(initial)
        self.initial = initial
        self.initial_improved = copy.deepcopy(initial)
        self.initial_improved['number_taxis'] = len(initial['taxis'].keys())
        self.initial_improved['number_passengers'] = len(initial['passengers'].keys())
        for taxi in initial['taxis'].keys():
            self.initial_improved['taxis'][taxi]['max_fuel'] = initial['taxis'][taxi]['fuel']
            self.initial_improved['taxis'][taxi]['max_capacity'] = initial['taxis'][taxi]['capacity']
        self.taxis_names = [taxi_name for taxi_name in initial['taxis'].keys()]
        self.possible_destination = self.possible_goals()


    def act(self, state):
        turns_to_go = state.pop('turns to go', None)
        taxis_fuels = {}
        taxis_capacity = {}
        for taxi_name in self.taxis_names:
            taxis_fuels[taxi_name] = state['taxis'][taxi_name].pop('fuel')
            taxis_capacity[taxi_name] = state['taxis'][taxi_name].pop('capacity')
        all_possible_actions = self.create_randoms_action()
        actions = self.actions(state, all_possible_actions)
        max_reward = -100
        max_action = None
        # noinspection PyTypeChecker
        for action in actions:
            copy_state = copy.deepcopy(state)
            self.result(copy_state, action)
            reward, action_to_s_prime = self.reward_state_new(state, action, copy_state, turns_to_go, taxis_fuels, taxis_capacity)
            if reward >= max_reward:
                max_reward = reward
                max_action = action_to_s_prime
        state['turns to go'] = turns_to_go
        for taxi_name in self.taxis_names:
            state['taxis'][taxi_name]['fuel'] = taxis_fuels[taxi_name]
            state['taxis'][taxi_name]['capacity'] = taxis_capacity[taxi_name]
        return max_action



    def reward_state_new(self, state, action_taxis, state_prime, turns_to_go, taxis_fuels, taxis_capacity):
        reward = 1
        diameter = self.distances.diameter
        counter_s_prime = 0
        for i in range(len(self.possible_destination)):
            counter_passenger = 0
            pb = 1
            for passenger_name in state['passengers'].keys():
                state_prime['passengers'][passenger_name]['destination'] = self.possible_destination[counter_s_prime][counter_passenger]
                if state['passengers'][passenger_name]['destination'] == state_prime['passengers'][passenger_name]['destination']:
                    pb *= (1 - state['passengers'][passenger_name]['prob_change_goal'])
                else:
                    pb *= state['passengers'][passenger_name]['prob_change_goal']
                counter_passenger += 1
            counter_s_prime += 1

            if action_taxis == 'terminate':
                can_terminate = True
                for taxi_name in state['taxis'].keys():
                    for passenger in self.initial['passengers'].keys():
                        distance_to_goal = 0
                        if state_prime['passengers'][passenger]['location'] == taxi_name:
                            distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state_prime['passengers'][passenger]['destination'])
                            # If the passenger is not on the taxis -> move action
                            # The location of the passenger is his location
                        elif state['passengers'][passenger]['location'] not in self.taxis_names:
                            distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state_prime['passengers'][passenger]['location'])
                        if taxis_fuels[taxi_name] >= distance_to_goal and distance_to_goal <= turns_to_go:
                            can_terminate = False
                if not can_terminate:
                    reward += - 2 * diameter
                else:
                    reward = diameter / 5


            elif action_taxis == 'reset':
                can_reset = True
                for taxi_name in state['taxis'].keys():
                    for passenger in self.initial['passengers'].keys():
                        distance_to_goal = 0
                        if state_prime['passengers'][passenger]['location'] == taxi_name:
                            distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state_prime['passengers'][passenger]['destination'])
                            # If the passenger is not on the taxis -> move action
                            # The location of the passenger is his location
                        elif state['passengers'][passenger]['location'] not in self.taxis_names:
                            distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state_prime['passengers'][passenger]['location'])
                        if taxis_fuels[taxi_name] != 0 and taxis_fuels[
                            taxi_name] >= distance_to_goal and distance_to_goal <= turns_to_go:
                            can_reset = False
                if not can_reset:
                    reward += -diameter
                else:
                    reward = diameter / 4
                if turns_to_go <= 10:
                    reward = -5 * diameter

            else:
                for action in action_taxis:
                    act = action[0]
                    taxi_name = action[1]
                    reward += self.initial_improved['taxis'][taxi_name]['max_capacity'] - taxis_capacity[taxi_name]
                    move_reward = 0
                    count = 0
                    bonus = 0
                    for passenger in self.initial['passengers'].keys():
                        if act == 'move':
                            # If the passenger is on the taxi -> move action
                            # The location of the passenger is the taxi's name
                            if state_prime['passengers'][passenger]['location'] == taxi_name:
                                if taxis_capacity[taxi_name] == 0:
                                    bonus = diameter - self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'], state['passengers'][passenger]['destination'])
                                count += 1
                                move_temp = bonus  + (diameter - self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'], state['passengers'][passenger]['destination']))
                                if move_temp >= move_reward:
                                    move_reward = move_temp
                            # If the passenger is not on the taxis -> move action
                            # The location of the passenger is his location
                            elif state['passengers'][passenger]['location'] not in self.taxis_names:
                                if state['passengers'][passenger]['location'] == state['passengers'][passenger]['destination']:
                                    bonus = - (diameter - self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state['passengers'][passenger]['location']))/2
                                count += 1
                                move_temp = bonus + (diameter - self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state['passengers'][passenger]['location']))
                                if move_temp >= move_reward:
                                    move_reward = move_temp
                            if count == len(self.initial['passengers'].keys()):
                                reward += 2*move_reward
                            if taxis_fuels[taxi_name] == 0:
                                reward += -500

                        elif act == 'drop off':
                            pass_name = action[2]
                            if taxis_capacity[taxi_name] < self.initial_improved['taxis'][taxi_name]['max_capacity']:
                                if state['passengers'][pass_name]['location'] == taxi_name:
                                    if state['passengers'][pass_name]['destination'] == state['taxis'][taxi_name]['location']:
                                        reward += 100
                            else:
                                reward += -500


                        elif act == 'pick up':
                            if taxis_capacity[taxi_name] <= 0:
                                reward -= 500
                            else:
                                reward += 50

                        elif act == 'refuel':
                            if taxis_fuels[taxi_name] == self.initial_improved['taxis'][taxi_name]['max_fuel']:
                                reward += -100
                            else:
                                distance_to_goal = -1
                                # If the passenger is on the taxi -> refuel action
                                # The location of the passenger is the taxi's name
                                if state['passengers'][passenger]['location'] == taxi_name:
                                    distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,state_prime['taxis'][taxi_name]['location'],state_prime['passengers'][passenger]['destination'])

                                # If the passenger is not on the taxis -> refuel action
                                # The location of the passenger is his location
                                elif state['passengers'][passenger]['location'] not in self.taxis_names:
                                    distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances, state['taxis'][taxi_name]['location'],state['passengers'][passenger]['location'])
                                if distance_to_goal >= taxis_fuels[taxi_name]:
                                    reward += 3 * diameter
                                else:
                                    reward += -diameter

                        elif act == 'wait':
                            if taxis_fuels[taxi_name] - 1> 0:
                                if state['passengers'][passenger]['location'] == taxi_name:
                                    if self.distances.check_distances(self.distances.shortest_path_distances,state['taxis'][taxi_name]['location'],state['passengers'][passenger]['destination']) == 1:
                                        almost_all_taxis = []
                                        for i in range(len(self.taxis_names)):
                                            if self.taxis_names[i] != taxi_name:
                                                almost_all_taxis.append(self.taxis_names[i])
                                        flag = True
                                        for i in almost_all_taxis:
                                            if self.distances.check_distances(self.distances.shortest_path_distances,state['taxis'][taxi_name]['location'],state['taxis'][i]['location']) == 1:
                                                flag = False
                                        if not flag:
                                            reward += 20
                                        else:
                                            reward += -10
                                    else:
                                        reward += -20
                                else:
                                    reward += -20
                            else:
                                reward += -20
            reward *= pb
        return reward, action_taxis


    def possible_goals(self):
        destination_goals_passengers = []
        for passenger_name in self.initial['passengers'].keys():
            destination_goal_passenger = [self.initial['passengers'][passenger_name]['possible_goals']]
            destination_goals_passengers += destination_goal_passenger
        return list(itertools.product(*destination_goals_passengers))



    def create_randoms_action(self):
        n = len(self.initial['map'])
        m = len(self.initial['map'][0])
        actions = []

        for taxi_name in self.initial['taxis'].keys():
            taxis_actions = []
            for i in range(n):
                for j in range(m):
                    taxis_actions.append(('move', taxi_name, (i, j)))
            for passenger_name in self.initial['passengers'].keys():
                taxis_actions.append(('pick up', taxi_name, passenger_name))
                taxis_actions.append(('drop off', taxi_name, passenger_name))
            taxis_actions.append(('refuel', taxi_name))
            taxis_actions.append(('wait', taxi_name))
            actions.append(taxis_actions)

        all_possible_actions = list(itertools.product(*actions))
        all_possible_actions.append('reset')
        all_possible_actions.append('terminate')
        return all_possible_actions

    def actions(self, state1, all_possible_actions):
        all_actions = []
        for possible_action in all_possible_actions:
            if self.is_action_legal(state1, possible_action):
                all_actions.append(possible_action)
        return all_actions

    def is_action_legal(self,state1, action):
        """
        check if the action is legal
        """
        def _is_move_action_legal(move_action):
            taxi_name = move_action[1]
            if taxi_name not in state1['taxis'].keys():
                return False
            # if self.state['taxis'][taxi_name]['fuel'] == 0:
            #     return False
            l1 = state1['taxis'][taxi_name]['location']
            l2 = move_action[2]
            return l2 in list(self.distances.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if state1['taxis'][taxi_name]['location'] != state1['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            # if state1['taxis'][taxi_name]['capacity'] <= 0:
            #     return False
            # check passenger is not in his destination
            if state1['passengers'][passenger_name]['destination'] == state1['passengers'][passenger_name]['location']:
                return False
            return True

        def _is_drop_action_legal(drop_action):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if state1['taxis'][taxi_name]['location'] == state1['passengers'][passenger_name]['destination']:
                if state1['passengers'][passenger_name]['location'] == taxi_name:
                    return True
            return False

        def _is_refuel_action_legal(refuel_action):
            """
            check if taxi in gas location
            """
            taxi_name = refuel_action[1]
            i, j = state1['taxis'][taxi_name]['location']
            if self.initial_improved['map'][i][j] == 'G':
                return True
            else:
                return False

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

        if action == "reset":
            return True
        if action == "terminate":
            return True

        if len(action) != len(state1["taxis"].keys()):
            return False

        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action):
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action):
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action):
                    return False
            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action):
                    return False
            elif atomic_action[0] != 'wait':
                return False

        # check mutex action
        if _is_action_mutex(action):
            return False
        # check taxis collision
        if len(state1['taxis']) > 1:
            taxis_location_dict = dict([(t, state1['taxis'][t]['location']) for t in state1['taxis'].keys()])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                return False
        return True

    def result(self, state1, action):
        """"
        update the state according to the action
        """
        self.apply(state1, action)
        if action != "reset":
            self.environment_step(state1)

    def apply(self, state1, action):
        """
        apply the action to the state
        """
        if action == "reset":
            self.reset_environment(state1)
            return
        if action == "terminate":
            self.terminate_execution(state1)
            return
        for atomic_action in action:
            self.apply_atomic_action(state1, atomic_action)

    def apply_atomic_action(self, state1, atomic_action):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state1['taxis'][taxi_name]['location'] = atomic_action[2]
            # state1['taxis'][taxi_name]['fuel'] -= 1
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            # state1['taxis'][taxi_name]['capacity'] -= 1
            state1['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            state1['passengers'][passenger_name]['location'] = state1['taxis'][taxi_name]['location']
            #####################
            # if state1['taxis'][taxi_name]['capacity'] < self.initial_improved['taxis'][taxi_name]['max_capacity']:
            #     state1['taxis'][taxi_name]['capacity'] += 1
            #####################
            return
        elif atomic_action[0] == 'refuel':
            state1['taxis'][taxi_name]['fuel'] = self.initial_improved['taxis'][taxi_name]['fuel']
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def environment_step(self, state1):
        """
        update the state of environment randomly
        """
        for p in state1['passengers']:
            passenger_stats = state1['passengers'][p]
            if random.random() < passenger_stats['prob_change_goal']:
                # change destination
                passenger_stats['destination'] = random.choice(passenger_stats['possible_goals'])
        # state1["turns to go"] -= 1
        return

    def reset_environment(self, state1):
        """
        reset the state of the environment
        """
        state1["taxis"] = copy.deepcopy(self.initial["taxis"])
        state1["passengers"] = copy.deepcopy(self.initial["passengers"])
        # state1["turns to go"] -= 1
        return

    def terminate_execution(self, state1):
        """
        terminate the execution of the problem
        """
        # print(f"End of game")
        # print(f"-----------------------------------")
        return

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
        g.graph['gaStation'] = []
        for node in g:
            if initial['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
            if initial['map'][node[0]][node[1]] == 'G':
                g.graph['gaStation'].append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        self.gaStation = g.graph['gaStation']
        return g

    def create_shortest_path_distances(self, graph):
        d = {}
        for n1 in graph.nodes:
            for n2 in graph.nodes:
                if n1 == n2:
                    continue
                d[(n1, n2)] = len(nx.shortest_path(graph, n1, n2)) - 1
        return d


    def check_distances(self, graph, node1, node2):
        if (node1, node2) not in graph:
            return 0
        else:
            if node1 == node2:
                return 0
            else:
                return graph[(node1, node2)]

