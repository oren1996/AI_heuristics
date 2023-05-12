def act(self, state):
    turns_to_go = state.pop('turns to go', None)
    taxis_fuels = {}
    taxis_capacity = {}
    for taxi_name in self.taxis_names:
        taxis_fuels[taxi_name] = state['taxis'][taxi_name].pop('fuel')
        taxis_capacity[taxi_name] = state['taxis'][taxi_name].pop('capacity')
    idx = self.lst_states.index(state)
    action_poss = self.lst_actions[idx]
    max_reward = -100
    max_action = None
    # noinspection PyTypeChecker
    for action in action_poss:
        reward, action_to_s_prime = self.reward_state_new(state, action, action_poss[action], turns_to_go, taxis_fuels,
                                                          taxis_capacity)
        if reward >= max_reward:
            max_reward = reward
            max_action = action_to_s_prime
    self.last_action = max_action
    self.last_state = state
    state['turns to go'] = turns_to_go
    for taxi_name in self.taxis_names:
        state['taxis'][taxi_name]['fuel'] = taxis_fuels[taxi_name]
        state['taxis'][taxi_name]['capacity'] = taxis_capacity[taxi_name]

    return max_action


def action_for_each_state(self):
    state_action = [0] * len(self.lst_states)
    all_possible_actions = self.create_randoms_action()
    counter_state = 0
    for state in self.lst_states:
        # print(counter_state)
        if counter_state % 500 == 0:
            print('-----')
            print(f'{counter_state} of {len(self.lst_states)}')

        actions = self.actions(state, all_possible_actions)
        actions_states = dict()
        for action in actions:
            # for each action, compute all the states you can go to
            number_of_possible_states = len(self.possible_destination)
            copy_state = copy.deepcopy(state)
            self.result(copy_state, action)
            s_primes = []
            for i in range(number_of_possible_states):
                pass_counter = 0
                copy_state_prime = copy.deepcopy(copy_state)
                for passenger_name in (state['passengers'].keys()):
                    copy_state_prime['passengers'][passenger_name]['destination'] = self.possible_destination[i][
                        pass_counter]
                    pass_counter += 1
                pb = self.transition_function(state, action, copy_state_prime)
                s_primes.append((copy_state_prime, pb))
            actions_states[action] = s_primes
            # action_state = { action: s_prime, action2:s_prime2
            # actions = { ( ('move', 'taxi1', (1,2) ), ('move', 'taxi1', (1,2)) ): 38, ..

        # noinspection PyTypeChecker
        state_action[counter_state] = actions_states
        counter_state += 1
    return state_action


def reward_state_new(self, state, action_taxis, state_prime_lst, turns_to_go, taxis_fuels, taxis_capacity):
    # state = {'optimal': False, 'map': [['P', 'P', 'P', 'P'], ['I', 'I', 'I', 'G'], ['P', 'P', 'P', 'P']], 'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 8, 'capacity': 1}}, 'passengers': {'Dana': {'location': (2, 0), 'destination': (0, 0), 'possible_goals': ((0, 0), (2, 0)), 'prob_change_goal': 0.01}}}
    # action_taxis = (('move', 'taxi 1', (0, 1)),)
    # state_prime_lst =  [({'optimal': False, 'map': [['P', 'P', 'P', 'P'], ['I', 'I', 'I', 'G'], ['P', 'P', 'P', 'P']], 'taxis': {'taxi 1': {'location': (0, 1), 'fuel': 7, 'capacity': 1}}, 'passengers': {'Dana': {'location': (2, 0), 'destination': (0, 0), 'possible_goals': ((0, 0), (2, 0)), 'prob_change_goal': 0.01}}}, 0.99), ({'optimal': False, 'map': [['P', 'P', 'P', 'P'], ['I', 'I', 'I', 'G'], ['P', 'P', 'P', 'P']], 'taxis': {'taxi 1': {'location': (0, 1), 'fuel': 7, 'capacity': 1}}, 'passengers': {'Dana': {'location': (2, 0), 'destination': (2, 0), 'possible_goals': ((0, 0), (2, 0)), 'prob_change_goal': 0.01}}}, 0.01)]

    reward = 1
    diameter = self.distances.diameter
    for i in range(len(state_prime_lst)):
        state_prime = state_prime_lst[i][0]
        pb = state_prime_lst[i][1]
        if action_taxis == 'terminate':
            can_terminate = True
            for taxi_name in state['taxis'].keys():
                for passenger in self.initial['passengers'].keys():
                    distance_to_goal = 0
                    if state_prime['passengers'][passenger]['location'] == taxi_name:
                        distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,
                                                                          state_prime['taxis'][taxi_name]['location'],
                                                                          state_prime['passengers'][passenger][
                                                                              'destination'])
                        # If the passenger is not on the taxis -> move action
                        # The location of the passenger is his location
                    elif state['passengers'][passenger]['location'] not in self.taxis_names:
                        distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,
                                                                          state_prime['taxis'][taxi_name]['location'],
                                                                          state_prime['passengers'][passenger][
                                                                              'location'])
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
                        distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,
                                                                          state_prime['taxis'][taxi_name]['location'],
                                                                          state_prime['passengers'][passenger][
                                                                              'destination'])
                        # If the passenger is not on the taxis -> move action
                        # The location of the passenger is his location
                    elif state['passengers'][passenger]['location'] not in self.taxis_names:
                        distance_to_goal = self.distances.check_distances(self.distances.shortest_path_distances,
                                                                          state_prime['taxis'][taxi_name]['location'],
                                                                          state_prime['passengers'][passenger][
                                                                              'location'])

                    if taxis_fuels[taxi_name] != 0 and taxis_fuels[
                        taxi_name] >= distance_to_goal and distance_to_goal <= turns_to_go:
                        can_reset = False
            if not can_reset:
                reward += -diameter
            else:
                reward = diameter / 2
            if turns_to_go <= 10:
                reward = -5 * diameter

        else:
            for action in action_taxis:
                act = action[0]
                taxi_name = action[1]
                # fuel_conservation = state['taxis'][taxi_name]['fuel'] / self.initial_improved['taxis'][taxi_name]['max_fuel']
                reward += self.initial_improved['taxis'][taxi_name]['max_capacity'] - taxis_capacity[taxi_name]
                for passenger in self.initial['passengers'].keys():
                    if act == 'move':
                        # If the passenger is on the taxi -> move action
                        # The location of the passenger is the taxi's name
                        if state_prime['passengers'][passenger]['location'] == taxi_name:
                            reward += diameter - self.distances.check_distances(self.distances.shortest_path_distances,
                                                                                state_prime['taxis'][taxi_name][
                                                                                    'location'],
                                                                                self.initial['passengers'][passenger][
                                                                                    'destination'])
                        # If the passenger is not on the taxis -> move action
                        # The location of the passenger is his location
                        elif state['passengers'][passenger]['location'] not in self.taxis_names:
                            reward += diameter - self.distances.check_distances(self.distances.shortest_path_distances,
                                                                                state_prime['taxis'][taxi_name][
                                                                                    'location'],
                                                                                self.initial['passengers'][passenger][
                                                                                    'location'])

                        if taxis_fuels[taxi_name] == 0:
                            reward += -500

                    elif act == 'drop off':
                        if taxis_capacity[taxi_name] < self.initial_improved['taxis'][taxi_name]['max_capacity']:
                            reward += 100
                        else:
                            reward += -500

                    elif act == 'pick up':
                        if taxis_capacity[taxi_name] <= 0:
                            reward -= 500
                        else:
                            reward += 20

                    elif act == 'refuel':
                        if taxis_fuels[taxi_name] == self.initial_improved['taxis'][taxi_name]['max_fuel']:
                            reward += -100
                        else:
                            distance_to_goal = -1
                            # If the passenger is on the taxi -> refuel action
                            # The location of the passenger is the taxi's name
                            if state['passengers'][passenger]['location'] == taxi_name:
                                distance_to_goal = self.distances.check_distances(
                                    self.distances.shortest_path_distances, state_prime['taxis'][taxi_name]['location'],
                                    state_prime['passengers'][passenger]['destination'])

                            # If the passenger is not on the taxis -> refuel action
                            # The location of the passenger is his location
                            elif state['passengers'][passenger]['location'] not in self.taxis_names:
                                distance_to_goal = self.distances.check_distances(
                                    self.distances.shortest_path_distances, state['taxis'][taxi_name]['location'],
                                    state['passengers'][passenger]['location'])
                            if distance_to_goal >= taxis_fuels[taxi_name]:
                                reward += 3 * diameter
                            else:
                                reward += -diameter

                    elif act == 'wait':
                        if taxis_fuels[taxi_name] - 1 > 0:
                            reward += -10
                        else:
                            reward += -20
        reward *= pb
    return reward, action_taxis




def value_state_initialization_new(self):
    state = self.initial_improved
    states = list()
    rewards = list()
    self.n = len(self.initial['map'])
    self.m = len(self.initial['map'][0])
    self.taxis_names = [taxi_name for taxi_name in state['taxis'].keys()]
    number_taxis = state['number_taxis']
    # capacity_taxis = []
    possible_location_taxis = number_taxis * [[(i, j) for i in range(self.n) for j in range(self.m) if self.initial['map'][i][j] != 'I']]
    possible_location_passengers = []


    possible_destination_passengers = []
    for passenger_name in state['passengers'].keys():
        poss_location = list()
        poss_destination = list()
        for possible_destination in state['passengers'][passenger_name]['possible_goals']:
            poss_location.append(possible_destination)
            poss_destination.append(possible_destination)
        poss_location += self.taxis_names
        if state['passengers'][passenger_name]['location'] not in poss_location:
            poss_location.append(state['passengers'][passenger_name]['location'])
        if state['passengers'][passenger_name]['destination'] not in poss_destination:
            poss_location.append(state['passengers'][passenger_name]['destination'])
            poss_destination.append(state['passengers'][passenger_name]['destination'])
        possible_location_passengers.append(poss_location)
        possible_destination_passengers.append(poss_destination)


    # for taxi_name in state['taxis'].keys():
    #     capacity_taxis += [[i for i in range(state['taxis'][taxi_name]['max_capacity'] + 1)]]

    all_possible_location_taxis = list(itertools.product(*possible_location_taxis))
    all_possible_location_passengers = list(itertools.product(*possible_location_passengers))
    all_possible_destination_passengers = list(itertools.product(*possible_destination_passengers))
    # all_capacity = list(itertools.product(*capacity_taxis))

    for location_taxis in all_possible_location_taxis:
        for location_passengers in all_possible_location_passengers:
            for destination_passengers in all_possible_destination_passengers:
                # for taxis_capacity in all_capacity:
                taxi_counter = 0
                passenger_counter = 0
                state_i = dict()
                state_i['optimal'] = state['optimal']
                state_i['map'] = state['map']
                state_i['taxis'] = {}
                state_i['passengers'] = {}

                for taxi_name in state['taxis'].keys():
                    state_i['taxis'][taxi_name] = {}
                    state_i['taxis'][taxi_name]['names_passengers_aboard'] = []
                    state_i['taxis'][taxi_name]["location"] = location_taxis[taxi_counter]
                    # state_i['taxis'][taxi_name]["capacity"] = taxis_capacity[taxi_counter]

                    taxi_counter += 1

                for passenger_name in state['passengers'].keys():
                    state_i['passengers'][passenger_name] = {}
                    state_i['passengers'][passenger_name]["location"] = location_passengers[passenger_counter]
                    state_i['passengers'][passenger_name]["destination"] = destination_passengers[passenger_counter]
                    state_i['passengers'][passenger_name]["possible_goals"] = state['passengers'][passenger_name]["possible_goals"]
                    state_i['passengers'][passenger_name]["prob_change_goal"] = state['passengers'][passenger_name]["prob_change_goal"]


                    if state_i['passengers'][passenger_name]['location'] in self.taxis_names :
                        taxi_name_idx = self.taxis_names.index(state_i['passengers'][passenger_name]['location'])
                        state_i['taxis'][self.taxis_names[taxi_name_idx]]['names_passengers_aboard'].append(passenger_name)

                    passenger_counter += 1

                flag_1 = False
                flag_2 = False
                flag_3 = False
                flag_4 = False

                for taxi_name in state_i['taxis'].keys():

                    if len(state_i['taxis'][taxi_name]['names_passengers_aboard']) > \
                            state['taxis'][taxi_name]['max_capacity']:
                        flag_1 = True
                    # if len(state_i['taxis'][taxi_name]['names_passengers_aboard']) == 0 and \
                    #         state_i['taxis'][taxi_name]['capacity'] == 0:
                    #     flag_2 = True
                    # if len(state_i['taxis'][taxi_name]['names_passengers_aboard']) != 0 and \
                    #         state_i['taxis'][taxi_name]['capacity'] == state['taxis'][taxi_name]['max_capacity']:
                    #     flag_3 = True
                    del state_i['taxis'][taxi_name]['names_passengers_aboard']

                if number_taxis == 1:
                    flag_4 = False

                elif number_taxis == 2:
                    if location_taxis[0] == location_taxis[1] and location_taxis[0] != self.initial['taxis'][self.taxis_names[0]]['location']:
                        flag_4 = True
                    else:
                        flag_4 = False
                else:
                    flag_4 = False

                # if not flag_1 and not flag_2 and not flag_3:
                if not flag_1 and not flag_4:
                    states.append(state_i)
                    rewards.append(0)

    self.possible_destination = self.possible_goals()
    return states, rewards



