import copy
import itertools

import networkx as nx

import search
import random
import math
import json
import utils
import time
# import networkx as nx
import logging
# import numpy as np
import statistics

ids = ["206202384", "204864532"]


def distance_with_I(map, a, b):
    x_a = a[0]
    y_a = a[1]
    x_b = b[0]
    y_b = b[1]

    one_side_dist = 0
    second_side_dist = 0
    one_side_list = []
    second_side_list = []
    if x_a > x_b and y_a < y_b:
        x = x_a
        y = y_a
        while y < y_b:
            y += 1
            one_side_list.append(map[x][y])
            one_side_dist += 1

        while x > x_b:
            x -= 1
            one_side_list.append(map[x][y])
            one_side_dist += 1


    elif x_a == x_b and y_a < y_b:
        x = x_a
        y = y_a
        while y < y_b:
            y += 1
            one_side_list.append(map[x][y])
            one_side_dist += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1



    elif x_a < x_b and y_a < y_b:
        x = x_a
        y = y_a
        while y < y_b:
            y += 1
            one_side_list.append(map[x][y])
            one_side_dist += 1

        while x < x_b:
            x += 1
            one_side_list.append(map[x][y])
            one_side_dist += 1


    elif x_a < x_b and y_a == y_b:
        x = x_a
        y = y_a
        while x < x_b:
            x += 1
            one_side_list.append(map[x][y])
            one_side_dist += 1


    elif x_a > x_b and y_a == y_b:
        x = x_a
        y = y_a
        while x > x_b:
            x -= 1
            one_side_list.append(map[x][y])
            one_side_dist += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1



    elif x_a > x_b and y_a > y_b:
        x = x_a
        y = y_a
        while y > y_b:
            y -= 1
            one_side_list.append(map[x][y])
            one_side_dist += 1
        while x > x_b:
            x -= 1
            one_side_list.append(map[x][y])
            one_side_dist += 1

    elif x_a == x_b and y_a > y_b:
        y = y_a
        x = x_a
        while y > y_b:
            y -= 1
            one_side_list.append(map[x][y])
            one_side_dist += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1

    elif x_a < x_b and y_a > y_b:
        x = x_a
        y = y_a
        while y > y_b:
            y -= 1
            one_side_list.append(map[x][y])
            one_side_dist += 1

        while x < x_b:
            x += 1
            one_side_list.append(map[x][y])
            one_side_dist += 1

    # check second side
    if x_a > x_b and y_a < y_b:
        x = x_a
        y = y_a
        while x > x_b:
            x -= 1
            second_side_list.append(map[x][y])
            second_side_dist += 1

        while y < y_b:
            y += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1


    elif x_a < x_b and y_a < y_b:
        x = x_a
        y = y_a
        while x < x_b:
            x += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1

        while y < y_b:
            y += 1
            second_side_list.append(map[x_b][y])
            second_side_dist += 1


    elif x_a < x_b and y_a == y_b:
        x = x_a
        y = y_a
        while x < x_b:
            x += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1


    elif x_a > x_b and y_a > y_b:
        x = x_a
        y = y_a
        while x > x_b:
            x -= 1
            second_side_list.append(map[x][y])
            second_side_dist += 1
        while y > y_b:
            y -= 1
            second_side_list.append(map[x][y])
            second_side_dist += 1

    elif x_a < x_b and y_a > y_b:
        x = x_a
        y = y_a
        while x < x_b:
            x += 1
            second_side_list.append(map[x][y])
            second_side_dist += 1

        while y > y_b:
            y -= 1
            second_side_list.append(map[x][y])
            one_side_dist += 1

    # check if there is I on the way
    for i in one_side_list:
        if i == 'I':
            one_side_dist += 2
            break

    for i in second_side_list:
        if i == 'I':
            second_side_dist += 2
            break

    return min(one_side_dist, second_side_dist)


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def euclidean(a, b):
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""

        count_nb_passengers = 0
        initial['names_passengers'] = []
        for passenger_name in initial['passengers']:
            initial['passengers'][passenger_name]['picked_up'] = False
            initial['passengers'][passenger_name]['dropped_off'] = False
            initial['names_passengers'].append(passenger_name)
            count_nb_passengers += 1

        for taxi in list(initial['taxis'].keys()):
            initial['taxis'][taxi]['max_fuel'] = initial['taxis'][taxi]['fuel']
            initial['taxis'][taxi]['max_capacity'] = initial['taxis'][taxi]['capacity']
            initial['taxis'][taxi]['names_passengers_aboard'] = []
            initial['taxis'][taxi]['ever_fueled'] = False

        initial['total_number_passengers'] = count_nb_passengers
        initial['goal_test_counter'] = 0
        initial['num_rows'] = len(initial['map'])
        initial['num_cols'] = len(initial['map'][0])
        # initial['rounds'] = 0
        initial['number_of_taxis'] = len(initial['taxis'])
        initial['number_passengers_picked_up'] = 0

        state = json.dumps(initial)
        # build new initial dictionary with new props
        search.Problem.__init__(self, state)

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        # return tuple of tuples of actions for each taxi
        # loop over all taxis, check every action for each one and save all the possible
        # combinations (cartesian multiplication).

        # length of all_actions is the number of taxis

        # NOTE: check what the actions should look a like. i.e. is it ok to be a list
        # which each entry of it is a list contains tuples of actions of a taxi?

        # state = copy.deepcopy(state)
        state = json.loads(state)
        all_actions = []
        # i.e. taxis = [('taxi 1', {'location': (3, 3), 'fuel': 15, 'capacity': 2})]
        for taxi in state['taxis'].items():
            temp = self.check_actions(taxi, state)
            all_actions.append(tuple(temp))
            # all_actions = [(('Move', 'Taxi1', (1,2)), ('wait', 'Taxi1'), ('Move', 'Taxi1', (0,0))), (('Move', 'Taxi2', (0,0)), ('wait', 'Taxi2'))]

        all_actions = tuple(all_actions)
        # all_actions = ((('Move', 'Taxi1', (1,2)), ('wait', 'Taxi1'), ('Move', 'Taxi1', (0,0))), (('Move', 'Taxi2', (0,0)), ('wait', 'Taxi2')))

        all_actions = list(itertools.product(*all_actions))
        if state['number_of_taxis'] == 1:
            state = json.dumps(state)
            for act in all_actions:
                yield act
        else:
            all_actions_copy = copy.deepcopy(all_actions)
            for actions in all_actions:
                for act in actions:
                    if act[0] == 'move':
                        for act2 in actions:
                            # check if it's the same taxi
                            if act[1] == act2[1]:
                                continue
                            # first case:
                            # if taxi a and taxi b want to move to the same tile
                            if (act2[0] == 'wait' or act2[0] == 'refuel' or act2[0] == 'pick up' or act2[
                                0] == 'drop off') and act[2] == state["taxis"][act2[1]]["location"]:
                                all_actions_copy.remove(actions)
                                break
                            if act2[0] == 'move':
                                if act[2] == act2[2]:
                                    all_actions_copy.remove(actions)
                                    break
                    break
            state = json.dumps(state)
            for act in all_actions_copy:
                yield act
            # i.e. act = (('wait', 'taxi 1'),)

    def check_actions(self, taxi, state):
        '''
        return a list of tuples represents all possible actions for a taxi.
        i.e: [(“move”, “taxi 1”, (1, 2)), (“wait”, “taxi 1”), ("refuel", "taxi 1")]
        '''
        M = state['num_rows']
        N = state['num_cols']
        taxi_name = taxi[0]
        taxi_location = taxi[1]['location']
        taxi_fuel = taxi[1]['fuel']
        taxi_max_fuel = taxi[1]['max_fuel']
        taxi_capacity = taxi[1]['capacity']
        taxi_max_capacity = taxi[1]['max_capacity']
        x, y = taxi_location
        actions = []

        # passengers names on taxi[i] for checking drop off action
        passengers_aboard_taxi = taxi[1]['names_passengers_aboard']

        # add the wait action
        actions.append(('wait', taxi_name))

        # check for refuel
        if state['map'][x][y] == 'G' and state['taxis'][taxi_name]['fuel'] != state['taxis'][taxi_name]['max_fuel']:
            actions.append(('refuel', taxi_name))

        # check if out of fuel
        if taxi_fuel <= 0:
            return actions

        # check for move actions
        # can go up
        if x - 1 >= 0 and state['map'][x - 1][y] != 'I':
            actions.append(('move', taxi_name, (x - 1, y)))

        # can go down
        if x + 1 < M and state['map'][x + 1][y] != 'I':
            actions.append(('move', taxi_name, (x + 1, y)))

        # can go left
        if y - 1 >= 0 and state['map'][x][y - 1] != 'I':
            actions.append(('move', taxi_name, (x, y - 1)))

        # can go right
        if y + 1 < N and state['map'][x][y + 1] != 'I':
            actions.append(('move', taxi_name, (x, y + 1)))

        # check for pick-up actions
        # passengers_names_and_locations = [['Yossi', (0, 0)], ['Moshe', (3, 1)]]
        if taxi_capacity > 0:
            for passenger_name in state['passengers']:
                if (state['passengers'][passenger_name]['location'] == taxi_location) and (
                        state['passengers'][passenger_name]['picked_up'] == False):
                    actions.append(('pick up', taxi_name, passenger_name))

        # check for drop_off actions
        if taxi_capacity < taxi_max_capacity:
            for passenger_name in passengers_aboard_taxi:
                if (state['passengers'][passenger_name]['destination'] == taxi_location) and (
                        state['passengers'][passenger_name]['dropped_off'] == False):
                    actions.append(('drop off', taxi_name, passenger_name))

        return utils.shuffled(actions)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        # i.e. if the action tells us to move taxi 1 to (1,2) we need to update
        # the state accordingly- decrease the taxi fuel and update its location in the state dict.
        # when pick-up or drop-off update the capacity.

        # state = copy.deepcopy(state)
        state = json.loads(state)
        # check how it should look like if there is only 1 taxi
        for act in action:
            taxi_name = act[1]
            taxi_action = act[0]
            if len(act) == 2:
                if taxi_action == 'wait':
                    continue
                elif taxi_action == 'refuel':
                    state['taxis'][taxi_name]['fuel'] = state['taxis'][taxi_name]['max_fuel']
                    state['taxis'][taxi_name]['ever_fueled'] = True
            elif len(act) == 3:
                if taxi_action == 'move':
                    cords = act[2]
                    state['taxis'][taxi_name]['fuel'] -= 1
                    state['taxis'][taxi_name]['location'] = cords
                    if state['taxis'][taxi_name]['capacity'] != state['taxis'][taxi_name]['max_capacity']:
                        for passenger_name in state['taxis'][taxi_name]['names_passengers_aboard']:
                            state['passengers'][passenger_name]['location'] = cords
                elif taxi_action == 'pick up':
                    passenger_name = act[2]
                    state['taxis'][taxi_name]['capacity'] -= 1
                    state['passengers'][passenger_name]['picked_up'] = True
                    state['number_passengers_picked_up'] += 1
                    state['taxis'][taxi_name]['names_passengers_aboard'].append(passenger_name)
                    ###
                    # state['unpicked_passengers_list'].remove(passenger_name)
                    ###
                elif taxi_action == 'drop off':
                    passenger_name = act[2]
                    state['taxis'][taxi_name]['capacity'] += 1
                    state['passengers'][passenger_name]['dropped_off'] = True
                    state['goal_test_counter'] += 1
                    state['taxis'][taxi_name]['names_passengers_aboard'].remove(passenger_name)

        # state['rounds'] += 1
        return json.dumps(state)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        state = json.loads(state)
        if state['goal_test_counter'] == state['total_number_passengers']:
            state = json.dumps(state)

            return True
        state = json.dumps(state)
        return False

    def h(self, node):
        penalty = 0
        state = json.loads(node.state)

        cycles_penalty = 0
        if node.parent is not None:
            # if there is 1 taxi
            if len(node.action) == 1:
                if node.action[0] == 'move':
                    parent = node.parent
                    grand_parent = parent.parent
                    if parent is not None and grand_parent is not None:
                        grand_parent_state = json.loads(grand_parent.state)
                        taxi_name = grand_parent_state['taxis'].keys()
                        if grand_parent_state['taxis'][taxi_name]['location'] == state['taxis'][taxi_name]['location']:
                            cycles_penalty += 10

            else:
                # if there is more than 1 taxi
                for action in node.action:
                    if node.action[0] == 'move':
                        parent = node.parent
                        grand_parent = parent.parent
                        if parent is not None and grand_parent is not None:
                            grand_parent_state = json.loads(grand_parent.state)
                            taxi_name = action[1]
                            if grand_parent_state['taxis'][taxi_name]['location'] == state['taxis'][taxi_name]['location']:
                                cycles_penalty += 10


        number_of_unpicked = state['total_number_passengers'] - state['number_passengers_picked_up']
        number_of_not_dropped_off = state['total_number_passengers'] - state['goal_test_counter']
        list_passengers_not_dropped_off = []
        for passenger_name in state['passengers']:
            if (state['passengers'][passenger_name]['picked_up'] == True) and (
                    state['passengers'][passenger_name]['dropped_off'] == False):
                list_passengers_not_dropped_off.append(passenger_name)

        distance_passenger_from_dest = 0
        for passenger_name in list_passengers_not_dropped_off:
            current_location = state['passengers'][passenger_name]['location']
            destination = state['passengers'][passenger_name]['destination']
            distance_passenger_from_dest += distance_with_I(state['map'], current_location, destination)

        if number_of_not_dropped_off == 0:
            mean_distances_passenger_from_dest = 0
        else:
            mean_distances_passenger_from_dest = distance_passenger_from_dest / number_of_not_dropped_off

        list_unpicked_passengers = list(set(state['names_passengers']) - set(list_passengers_not_dropped_off))
        distance_passenger_from_taxi = 0
        for passenger_name in list_unpicked_passengers:
            passenger_location = state['passengers'][passenger_name]['location']
            distance_passenger_from_taxi += self.closest_taxi(state, passenger_location)

        if number_of_unpicked == 0:
            mean_distances_passenger_from_taxi = 0  # maybe a negative number
        else:
            mean_distances_passenger_from_taxi = distance_passenger_from_taxi / number_of_unpicked

        closest_passenger_aboard = 0
        for taxi_name in list(state['taxis'].keys()):
            closest_passenger_aboard = self.closest_passenger(state, taxi_name)
            if closest_passenger_aboard < state['taxis'][taxi_name]['fuel'] and closest_passenger_aboard < 0 and not state['taxis'][taxi_name]['ever_fueled']:
                penalty -= 5

        if number_of_not_dropped_off == 1 and state['number_of_taxis'] == 1 and list_passengers_not_dropped_off:
            passenger_name = list_passengers_not_dropped_off[0]
            passenger_destination = state['passengers'][passenger_name]['destination']
            taxi_name = list(state['taxis'].keys())[0]
            taxi_location = state['taxis'][taxi_name]['location']
            dist_taxi_destination = distance_with_I(state['map'], taxi_location, passenger_destination)
            if dist_taxi_destination <= state['taxis'][taxi_name]['fuel']:
                penalty -= 2


        penalty += (number_of_unpicked * 10) + (number_of_not_dropped_off) + mean_distances_passenger_from_dest + mean_distances_passenger_from_taxi + cycles_penalty
        return penalty

    def closest_passenger(self, state, taxi_name):
        min_dist = []
        for passenger_aboard in state['taxis'][taxi_name]['names_passengers_aboard']:
            passenger_aboard_destination = state['passengers'][passenger_aboard]['destination']
            taxi_location = state['taxis'][taxi_name]['location']
            dist = distance_with_I(state['map'], taxi_location, passenger_aboard_destination)
            min_dist.append(dist)
        if not min_dist:
            return 0
        return min(min_dist)

    def closest_taxi(self, state, passenger_location):
        distances = []
        for taxi_name in list(state['taxis'].keys()):
            taxi_location = state['taxis'][taxi_name]['location']
            dist = distance_with_I(state['map'], taxi_location, passenger_location)
            distances.append(dist)
        return min(distances)

    def h_0(self, node):
        return 0

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = json.loads(node.state)
        number_unpicked = state['total_number_passengers'] - state['number_passengers_picked_up']
        number_picked_undelivered = state['number_passengers_picked_up'] - state['goal_test_counter']
        number_taxis_in_problem = state['number_of_taxis']
        return ((2 * number_unpicked) + number_picked_undelivered) / number_taxis_in_problem

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state = json.loads(node.state)

        return self.use_manhattan(state, penalty=0)

    def h_euc(self, node):
        """
        This is a slightly more sophisticated euclidian heuristic
        """
        state = json.loads(node.state)
        D = []
        T = []

        for passenger in state['passengers']:
            if state['passengers'][passenger]['picked_up'] == False and state['passengers'][passenger][
                'dropped_off'] == False:
                passenger_location = state['passengers'][passenger]['location']
                passenger_destination = state['passengers'][passenger]['destination']
                D.append(euclidean(passenger_location, passenger_destination))
            if state['passengers'][passenger]['picked_up'] == True and state['passengers'][passenger][
                'dropped_off'] == False:
                passenger_location = state['passengers'][passenger]['location']
                passenger_destination = state['passengers'][passenger]['destination']
                T.append(euclidean(passenger_location, passenger_destination))

        return (sum(D) + sum(T)) / state['number_of_taxis']

    def use_manhattan(self, state, penalty):
        D = []
        T = []

        for passenger in state['passengers']:
            if state['passengers'][passenger]['picked_up'] == False and state['passengers'][passenger][
                'dropped_off'] == False:
                passenger_location = state['passengers'][passenger]['location']
                passenger_destination = state['passengers'][passenger]['destination']
                D.append(manhattan(passenger_location, passenger_destination))
            if state['passengers'][passenger]['picked_up'] == True and state['passengers'][passenger][
                'dropped_off'] == False:
                passenger_location = state['passengers'][passenger]['location']
                passenger_destination = state['passengers'][passenger]['destination']
                T.append(manhattan(passenger_location, passenger_destination))

        return ((sum(D) + sum(T)) / state['number_of_taxis']) + penalty

    def use_manhattan_with_I(self, state, penalty):
        D = []
        T = []
        map = state['map']

        for passenger in state['passengers']:
            if state['passengers'][passenger]['picked_up'] == False and state['passengers'][passenger][
                'dropped_off'] == False:
                passenger_location = state['passengers'][passenger]['location']
                passenger_destination = state['passengers'][passenger]['destination']
                D.append(distance_with_I(map, passenger_location, passenger_destination))
            if state['passengers'][passenger]['picked_up'] == True and state['passengers'][passenger][
                'dropped_off'] == False:
                passenger_location = state['passengers'][passenger]['location']
                passenger_destination = state['passengers'][passenger]['destination']
                T.append(distance_with_I(map, passenger_location, passenger_destination))

        return ((sum(D) + sum(T)) / state['number_of_taxis']) + penalty


def create_taxi_problem(game):
    return TaxiProblem(game)
