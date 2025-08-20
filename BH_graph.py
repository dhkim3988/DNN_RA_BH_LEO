import numpy as np
import matplotlib.pyplot as plt

import Train_test_module

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
R_min = 0.1 # minimum data requirement (MHz)
average_demand = 1.35 # average traffic demand

reuse = 0  # 0 is first training, 1 is after training
keep_train = 0 # 0 only show test result, 1 can activate training
Heuristic_scheme_val = 0
# 0 Not using
# 1 is conventional iterative algorithm
# 2 is conventional per-slot algorithm
# 3 is IPM-based power allocation algorithm

DL_scheme_val = 0
# 0 Not using
# 1 is unsupervised learning-based timeslot allocation algorithm

# 2 is DQN-based timeslot allocation algorithm (timeslot only) [Full channel]
# 3 is DQN-based timeslot allocation algorithm (graph) [Full channel]
# 4 is unsupervised learning-based power allocation algorithm [Full channel]
# 5 is unsupervised learning-based power allocation algorithm [Full channel] (capacity estimation)

# 6 is DQN-based timeslot allocation algorithm (timeslot only) [Partial channel]
# 7 is DQN-based timeslot allocation algorithm (graph) [Partial channel]
# 8 is unsupervised learning-based power allocation algorithm [Partial channel]

num_train_instances = 10000
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

TEST_VALUE = 3
# 0: Average demand variation (power allocation)
# 1: User variation (power allocation)
# 2: Training results
# 3: rho variation

if TEST_VALUE == 0:
    EE1 = []
    EE2 = []
    EE3 = []
    EE4 = []
    EE5 = []

    outage1 = []
    outage2 = []
    outage3 = []
    outage4 = []
    outage5 = []

    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []

    demand_dim = []

    print('###############################################################')
    print('Graph making (requested demand variation)')
    print('###############################################################')

    for average_demand in [1.15,1.2,1.25,1.3,1.35,1.4]:
        demand_dim.append(average_demand)

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
                ]  # checkpoint for proposed DNN-based schemes

        savefile_prop_heuristic = [
            './Iterative/power_allocation_val/' + savefile_str,
            './Iterative/time_allocation_val/' + savefile_str,
            './Iterative/computation_time/' + savefile_str
        ]

        savefile_conv_heuristic = [
            './Per_slot/time_allocation_val/' + savefile_str,
            './Per_slot/computation_time/' + savefile_str
        ]

        Graph_result = Train_test_module.Train_test(Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                                                    dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                                                    test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor)

        PS_power_val, PS_EE, PS_outage, PS_time = Graph_result.graph_result(heuristic_scheme_val=3, DL_scheme_val=0)
        Iterative_power_val, Iterative_EE, Iterative_outage, Iterative_time = Graph_result.graph_result(heuristic_scheme_val=4, DL_scheme_val=0)
        IPM_power_val, IPM_EE, IPM_outage, IPM_time = Graph_result.graph_result(heuristic_scheme_val=6, DL_scheme_val=0)
        DNN_conv_power_val, DNN_conv_EE, DNN_conv_outage, DNN_conv_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=4)
        DNN_prop_power_val, DNN_prop_EE, DNN_prop_outage, DNN_prop_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=5)

        EE1.append(PS_EE)
        EE2.append(Iterative_EE)
        EE3.append(IPM_EE)
        EE4.append(DNN_conv_EE)
        EE5.append(DNN_prop_EE)

        outage1.append(PS_outage / 100.0)
        outage2.append(Iterative_outage / 100.0)
        outage3.append(IPM_outage / 100.0)
        outage4.append(DNN_conv_outage / 100.0)
        outage5.append(DNN_prop_outage / 100.0)

        time1.append(PS_time * 1000.0)
        time2.append(Iterative_time * 1000.0)
        time3.append(IPM_time * 1000.0)
        time4.append(DNN_conv_time * 1000.0)
        time5.append(DNN_prop_time * 1000.0)

        print('Requested traffic demand: ', average_demand)

    plt.plot(demand_dim, EE1, c='r', marker='v', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, EE2, c='black', ls='-.', marker='s', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, EE3, c='g', marker='^', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, EE4, c='b', marker='<', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, EE5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(EE1)
    print(EE2)
    print(EE3)
    print(EE4)
    print(EE5)

    plt.xticks(demand_dim)
    plt.grid()

    plt.xlabel('Average requested demand [Mbps]')
    plt.ylabel('Energy efficiency [Mbps/J]')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

    plt.plot(demand_dim, outage1, c='r', marker='v', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, outage2, c='black', ls='-.', marker='s', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, outage3, c='g', marker='^', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, outage4, c='b', marker='<', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, outage5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(outage1)
    print(outage2)
    print(outage3)
    print(outage4)
    print(outage5)

    plt.xticks(demand_dim)
    plt.grid()

    plt.xlabel('Average requested demand [Mbps]')
    plt.ylabel('Outage rate')
    plt.yscale('log')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

    plt.plot(demand_dim, time1, c='r', marker='v', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, time2, c='black', ls='-.', marker='s', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, time3, c='g', marker='^', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, time4, c='b', marker='<', ms='12', mew='1', mfc='w')
    plt.plot(demand_dim, time5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(time1)
    print(time2)
    print(time3)
    print(time4)
    print(time5)

    plt.xticks(demand_dim)
    plt.grid()

    plt.xlabel('Average requested demand [Mbps]')
    plt.ylabel('Computational time [msec]')
    plt.yscale('log')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

if TEST_VALUE == 1:
    EE1 = []
    EE2 = []
    EE3 = []
    EE4 = []
    EE5 = []

    outage1 = []
    outage2 = []
    outage3 = []
    outage4 = []
    outage5 = []

    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []

    user_dim = []

    print('###############################################################')
    print('Graph making (the number of users variation)')
    print('###############################################################')

    for num_user in [2,3,4,5,6]:
        total_user = num_user * num_fixed_beam
        user_dim.append(total_user)

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
                ]  # checkpoint for proposed DNN-based schemes

        savefile_prop_heuristic = [
            './Iterative/power_allocation_val/' + savefile_str,
            './Iterative/time_allocation_val/' + savefile_str,
            './Iterative/computation_time/' + savefile_str
        ]

        savefile_conv_heuristic = [
            './Per_slot/time_allocation_val/' + savefile_str,
            './Per_slot/computation_time/' + savefile_str
        ]

        Graph_result = Train_test_module.Train_test(Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                                                    dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                                                    test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor)

        PS_power_val, PS_EE, PS_outage, PS_time = Graph_result.graph_result(heuristic_scheme_val=3, DL_scheme_val=0)
        Iterative_power_val, Iterative_EE, Iterative_outage, Iterative_time = Graph_result.graph_result(heuristic_scheme_val=4, DL_scheme_val=0)
        #IPM_power_val, IPM_EE, IPM_outage, IPM_time = Graph_result.graph_result(heuristic_scheme_val=6, DL_scheme_val=0)
        DNN_conv_power_val, DNN_conv_EE, DNN_conv_outage, DNN_conv_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=4)
        DNN_prop_power_val, DNN_prop_EE, DNN_prop_outage, DNN_prop_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=5)

        EE1.append(PS_EE)
        EE2.append(Iterative_EE)
        #EE3.append(IPM_EE)
        EE4.append(DNN_conv_EE)
        EE5.append(DNN_prop_EE)

        outage1.append(PS_outage / 100.0)
        outage2.append(Iterative_outage / 100.0)
        #outage3.append(IPM_outage / 100.0)
        outage4.append(DNN_conv_outage / 100.0)
        outage5.append(DNN_prop_outage / 100.0)

        time1.append(PS_time * 1000.0)
        time2.append(Iterative_time * 1000.0)
        #time3.append(IPM_time * 1000.0)
        time4.append(DNN_conv_time * 1000.0)
        time5.append(DNN_prop_time * 1000.0)

        print('Total number of users: ', total_user)

    plt.plot(user_dim, EE1, c='r', marker='v', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, EE2, c='black', ls='-.', marker='s', ms='12', mew='1', mfc='w')
    #plt.plot(user_dim, EE3, c='g', marker='^', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, EE4, c='b', marker='<', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, EE5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(EE1)
    print(EE2)
    #print(EE3)
    print(EE4)
    print(EE5)

    plt.xticks(user_dim)
    plt.grid()

    plt.xlabel('The total number of users')
    plt.ylabel('Energy efficiency [Mbps/J]')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.xlim(user_dim[0], user_dim[-1])
    plt.tight_layout()

    plt.show()

    plt.plot(user_dim, outage1, c='r', marker='v', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, outage2, c='black', ls='-.', marker='s', ms='12', mew='1', mfc='w')
    #plt.plot(user_dim, outage3, c='g', marker='^', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, outage4, c='b', marker='<', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, outage5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(outage1)
    print(outage2)
    #print(outage3)
    print(outage4)
    print(outage5)

    plt.xticks(user_dim)
    plt.grid()

    plt.xlabel('The total number of users')
    plt.ylabel('Outage rate')
    plt.yscale('log')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.xlim(user_dim[0], user_dim[-1])
    plt.tight_layout()

    plt.show()

    plt.plot(user_dim, time1, c='r', marker='v', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, time2, c='black', ls='-.', marker='s', ms='12', mew='1', mfc='w')
    #plt.plot(user_dim, time3, c='g', marker='^', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, time4, c='b', marker='<', ms='12', mew='1', mfc='w')
    plt.plot(user_dim, time5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(time1)
    print(time2)
    #print(time3)
    print(time4)
    print(time5)

    plt.xticks(user_dim)
    plt.grid()

    plt.xlabel('The total number of users')
    plt.ylabel('Computational time [msec]')
    plt.yscale('log')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.xlim(user_dim[0], user_dim[-1])
    plt.tight_layout()

    plt.show()

if TEST_VALUE == 2:
    print('###############################################################')
    print('Graph making (training results)')
    print('###############################################################')

    average_demand = 42.0
    num_user = 3

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
            ]  # checkpoint for proposed DNN-based schemes

    savefile_prop_heuristic = [
        './Iterative/power_allocation_val/' + savefile_str,
        './Iterative/time_allocation_val/' + savefile_str,
        './Iterative/computation_time/' + savefile_str
    ]

    savefile_conv_heuristic = [
        './Per_slot/time_allocation_val/' + savefile_str,
        './Per_slot/computation_time/' + savefile_str
    ]

    Graph_result = Train_test_module.Train_test(Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                                                dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                                                test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor)

    PS_power_val, PS_EE, PS_outage, PS_time = Graph_result.graph_result(heuristic_scheme_val=3, DL_scheme_val=0)
    Iterative_power_val, Iterative_EE, Iterative_outage, Iterative_time = Graph_result.graph_result(heuristic_scheme_val=4, DL_scheme_val=0)
    IPM_power_val, IPM_EE, IPM_outage, IPM_time = Graph_result.graph_result(heuristic_scheme_val=6, DL_scheme_val=0)
    DNN_conv_power_val, DNN_conv_EE, DNN_conv_outage, DNN_conv_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=4)
    DNN_prop_power_val, DNN_prop_EE, DNN_prop_outage, DNN_prop_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=3)

    training_epochs = 24000

    ones_data = np.ones([int(training_epochs/10)])

    X_plot = np.array((range(1, int(training_epochs/10)+1))) * 10
    print(PS_EE)
    print(Iterative_EE)
    print(IPM_EE)
    print(DNN_conv_EE)

    plt.plot(X_plot, ones_data * PS_EE, c='r')
    plt.plot(X_plot, ones_data * Iterative_EE, c='black')
    plt.plot(X_plot, ones_data * IPM_EE, c='g')
    plt.plot(X_plot, ones_data * DNN_conv_EE, c='b')
    plt.plot(X_plot, DNN_prop_EE, c='indigo')

    plt.xticks(np.array(range(11)) * training_epochs / 10)
    plt.grid()

    plt.xlabel('The number of epochs')
    plt.ylabel('Energy efficiency [Mbps/J]')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    plt.tight_layout()

    plt.show()

    print(PS_outage)
    print(Iterative_outage)
    print(IPM_outage)
    print(DNN_conv_outage)

    plt.plot(X_plot, ones_data * PS_outage / 100.0, c='r')
    plt.plot(X_plot, ones_data * Iterative_outage / 100.0, c='black')
    plt.plot(X_plot, ones_data * IPM_outage / 100.0, c='g')
    plt.plot(X_plot, ones_data * DNN_conv_outage / 100.0, c='b')
    plt.plot(X_plot, DNN_prop_outage / 100.0, c='indigo')

    plt.xticks(np.array(range(11)) * training_epochs / 10)
    plt.grid()

    plt.xlabel('The number of epochs')
    plt.ylabel('Outage rate')
    plt.yscale('log')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    #plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

    print(10.0 * np.log10(PS_power_val))
    print(10.0 * np.log10(Iterative_power_val))
    print(10.0 * np.log10(IPM_power_val))
    print(10.0 * np.log10(DNN_conv_power_val))

    plt.plot(X_plot, ones_data * 10.0 * np.log10(PS_power_val), c='r')
    plt.plot(X_plot, ones_data * 10.0 * np.log10(Iterative_power_val), c='black')
    plt.plot(X_plot, ones_data * 10.0 * np.log10(IPM_power_val), c='g')
    plt.plot(X_plot, ones_data * 10.0 * np.log10(DNN_conv_power_val), c='b')
    plt.plot(X_plot, 10.0 * np.log10(DNN_prop_power_val), c='indigo')

    plt.xticks(np.array(range(11)) * training_epochs / 10)
    plt.grid()

    plt.xlabel('The number of epochs')
    plt.ylabel('Cost function [dBm]')
    plt.legend(['Per slot + TR-IPM', 'Iterative + TR-IPM', 'Prop. DQN + TR-IPM', 'Prop. DQN + Conv. DNN', 'Prop. DQN + Prop. DNN'])

    #plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

if TEST_VALUE == 3:
    EE1 = []
    EE2 = []
    EE3 = []
    EE4 = []
    EE5 = []

    outage1 = []
    outage2 = []
    outage3 = []
    outage4 = []
    outage5 = []

    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []

    demand_dim = []

    print('###############################################################')
    print('Graph making (rho variation)')
    print('###############################################################')

    for rho_val in [0.8,0.85,0.9,0.95,0.99,0.999]:
        average_demand = 1.4
        demand_dim.append(rho_val)
        num_user = 3

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
                ]  # checkpoint for proposed DNN-based schemes

        savefile_prop_heuristic = [
            './Iterative/power_allocation_val/' + savefile_str,
            './Iterative/time_allocation_val/' + savefile_str,
            './Iterative/computation_time/' + savefile_str
        ]

        savefile_conv_heuristic = [
            './Per_slot/time_allocation_val/' + savefile_str,
            './Per_slot/computation_time/' + savefile_str
        ]

        Graph_result = Train_test_module.Train_test(Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                                                    dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                                                    test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor)

        DNN_prop_power_val, DNN_prop_EE, DNN_prop_outage, DNN_prop_time = Graph_result.graph_result(heuristic_scheme_val=0, DL_scheme_val=5, rho_val=rho_val)

        EE5.append(DNN_prop_EE)
        outage5.append(DNN_prop_outage / 100.0)
        time5.append(DNN_prop_time * 1000.0)

        print('rho value: ', rho_val)

    plt.plot(demand_dim, EE5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(EE5)

    plt.xticks(demand_dim)
    plt.grid()

    plt.xlabel('rho value')
    plt.ylabel('Energy efficiency [Mbps/J]')
    plt.legend(['Prop. DQN + Prop. DNN'])

    plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

    plt.plot(demand_dim, outage5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(outage5)

    plt.xticks(demand_dim)
    plt.grid()

    plt.xlabel('rho value')
    plt.ylabel('Outage rate')
    plt.yscale('log')
    plt.legend(['Prop. DQN + Prop. DNN'])

    plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()

    plt.plot(demand_dim, time5, c='indigo', marker='o', ms='12', mew='1', mfc='w')

    print(time5)

    plt.xticks(demand_dim)
    plt.grid()

    plt.xlabel('Average requested demand [Mbps]')
    plt.ylabel('Computational time [msec]')
    plt.yscale('log')
    plt.legend(['Prop. DQN + Prop. DNN'])

    plt.xlim(demand_dim[0], demand_dim[-1])
    plt.tight_layout()

    plt.show()