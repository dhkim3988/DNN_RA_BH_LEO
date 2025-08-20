# KICS summer conference paper simulation code

import numpy as np
import tensorflow as tf
import time
import Train_test_module
import Deepexit as DE

Altitude = 600 # Altitude of LEO satellite (km)

num_fixed_beam = 16 # the number of fixed beams
num_active_beam = 4 # the number of active beams
num_group = int(np.ceil(num_fixed_beam / num_active_beam))
num_user = 3 # the number of users in each cell
num_timeslot = 32 # the number of timeslots
num_angle = 6 # the number of elevation angle

Earth_radius = 6371 # Radius of Earth (Km)
maximum_transmit_power = 53 # maximum transmit power of LEO satellite dBm
carrier_frequency = 2 # Carrier frequency of systems (GHz)
user_antenna_gain = 23 # ground user antenna gain (dB)
bandwidth = 10 # channel bandwidth (MHz)
dB3_angle = 3.0 # 3dB beamwidth angle (degree)
noise = 174 # noise spectral density (dBm/Hz)
elevation_angle_candidate = [40,50,60,70,80,90] # elevation angle candidate
R_min = 0.01 # minimum data requirement (MHz)
average_demand = 1.35 # average traffic demand

reuse = 0  # 0 is first training, 1 is after training
keep_train = 1 # 0 only show test result, 1 can activate training
Heuristic_scheme_val = 0
# 0 Not using
# 1 is conventional iterative algorithm
# 2 is conventional per-slot algorithm
# 3 is per-slot + IPM-based power allocation algorithm
# 4 is iterative + IPM-based power allocation algorithm
# 5 is DQN + IPM-based power allocation algorithm
# 6 is Dueling DQN + IPM-based power allocation algorithm

DL_scheme_val = 0
# 0 Not using
# 1 is unsupervised learning-based timeslot allocation algorithm
# 2 is DQN-based timeslot allocation algorithm (timeslot only) [Full channel]
# 3 is DQN-based timeslot allocation algorithm (graph) [Full channel]
# 4 is unsupervised learning-based power allocation algorithm [Full channel]
# 5 is proposed capacity estimation algorithm [Full channel]
# 6 is DQN-based timeslot allocation algorithm (dueling DQN) [Full channel]

num_train_instances = 4000
num_test_instances = 2000
train_seed = 0
test_seed = 17

# hyper parameters
learning_rate = [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.0001]
training_epochs = [0, 1000, num_train_instances*num_angle, num_train_instances*num_angle, 1000, 1000, num_train_instances*num_angle]
batch_size = [0, 10000, 32, 32, 1000*num_angle, 1000*num_angle, 32]
lambda_val = [0.0, 0.0, 0.0, 0.0, 0.0001, 0.002, 0.0] # Lagrangian multiplier for DNN-based schemes
# If sum rate maximization in DL_scheme_val = 5 -> learning rate 0.00001, lambda 0.002

# DQN parameters
epsilon = 1.0  # Initialization value of epsilon greedy
eps_decay = 0.995  # Epsilon decay value
eps_min = 0.01  # Minimum value of epsilon
discount_factor = 0.2  # Discount factor for future rewards

# num_user: [3,4,5,6,7] (demand 10.0)
# average_demand: [10.0,12.5,15.0,17.5,20.0] (user 3)

#[1.15,1.2,1.25,1.3,1.35,1.4]
#for num_user in [3]:
for average_demand in [1.35]:
    num_user = 3
    #for DL_scheme_val in [5]:
    for Heuristic_scheme_val in [6]:
        savefile_str = ('Altitude' + str(Altitude) + 'active_beam' + str(num_active_beam) +
                        'fixed_beam' + str(num_fixed_beam) + 'user' + str(num_user) + 'timeslot' + str(num_timeslot) +
                        'elevation_angle' + str(num_angle) + 'Tx_power' + str(maximum_transmit_power) +
                        'average_demand' + str(average_demand) + 'bandwidth' + str(bandwidth)
                        )

        ckpt = ['',
                './ckpt/unsupervised_timeslot_allocation/' + savefile_str + 'model.ckpt',
                './ckpt/proposed_DQN_timeslot/' + savefile_str + 'model.ckpt',
                './proposed_DQN_timeslot_graph/' + savefile_str,
                './ckpt/power_allocation/' + savefile_str + 'model.ckpt',
                './ckpt/power_allocation_capacity_estimation/' + savefile_str + 'model.ckpt',
                './ckpt/reward_optimization_DQN/' + savefile_str + 'model.ckpt'
                ] # checkpoint for proposed DNN-based schemes

        savefile_prop_heuristic = [
                        './Iterative/power_allocation_val/' + savefile_str,
                        './Iterative/time_allocation_val/' + savefile_str,
                        './Iterative/computation_time/' + savefile_str
                        ]

        savefile_conv_heuristic = [
                        './Per_slot/time_allocation_val/' + savefile_str,
                        './Per_slot/computation_time/' + savefile_str
                        ]

        Heuristic = Train_test_module.Train_test(Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                                                 dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                                                 test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor)
        Heuristic.heuristic_result()

        DNN = Train_test_module.Train_test(Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                                           dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                                           test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor)
        DNN.DNN_result()
