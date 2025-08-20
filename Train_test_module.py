import numpy as np
import time as time_module
import BH_channel
import BH_dnn
import scipy as sp
from itertools import product
import tensorflow as tf
from numba import cuda

class Train_test:
    def __init__(self, Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, reuse, keep_train, Heuristic_scheme_val, DL_scheme_val, num_train_instances, num_test_instances, train_seed,
                test_seed, learning_rate, training_epochs, batch_size, lambda_val, ckpt, savefile_conv_heuristic, savefile_prop_heuristic, epsilon, eps_decay, eps_min, discount_factor):
        self.Altitude = Altitude
        self.num_fixed_beam = num_fixed_beam
        self.num_active_beam = num_active_beam
        self.num_group = num_group
        self.num_user = num_user
        self.num_timeslot = num_timeslot
        self.num_angle = num_angle

        self.Earth_radius = Earth_radius
        self.maximum_transmit_power = maximum_transmit_power
        self.carrier_frequency = carrier_frequency
        self.user_antenna_gain = user_antenna_gain
        self.bandwidth = bandwidth
        self.dB3_angle = dB3_angle
        self.noise = noise
        self.elevation_angle_candidate = np.array(elevation_angle_candidate) * np.pi / 180.0
        self.R_min = R_min
        self.average_demand = average_demand

        self.reuse = reuse
        self.keep_train = keep_train
        self.Heuristic_scheme_val = Heuristic_scheme_val
        self.DL_scheme_val = DL_scheme_val

        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.train_seed = train_seed
        self.test_seed = test_seed
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lambda_val = lambda_val

        self.ckpt = ckpt
        self.savefile_conv_heuristic = savefile_conv_heuristic
        self.savefile_prop_heuristic = savefile_prop_heuristic

        self.num_train_data_conv = self.num_train_instances * self.num_timeslot * self.num_angle
        self.num_test_data_conv = self.num_test_instances * self.num_timeslot * self.num_angle

        self.num_train_data_prop = self.num_train_instances * self.num_angle
        self.num_test_data_prop = self.num_test_instances * self.num_angle

        self.total_user = self.num_user * self.num_fixed_beam

        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.discount_factor = discount_factor

    def DNN_result(self):
        # train step
        if self.DL_scheme_val == 0:
            pass

        if self.DL_scheme_val == 1:
            if self.keep_train == 1:
                print('###############################################################')
                print('Unsupervised Learning for Timeslot Allocation Training Process')
                print('Making channel sample : %d' % self.num_train_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)

                train_channel_gain, train_group_channel_gain = Train_Chn.Channel_gain()
                train_traffic_demand = Train_Chn.Demand_generation()
                train_snapshot_index, train_group_snapshot_index = Train_Chn.snapshot_strategy1()

                Train_DNN = BH_dnn.Beamhopping_train(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                     self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt, self.reuse)

                input_desired_signal, input_inter_signal, input_traffic_demand = Train_Chn.input_DNN(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)
                Train_DNN.Unsupervised_Timeslot_Allocation_train(input_desired_signal, input_inter_signal, input_traffic_demand, train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)

            print('###############################################################')
            print('Unsupervised Learning for Timeslot Allocation Testing Process')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                 self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                 self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt)

            input_desired_signal, input_inter_signal, input_traffic_demand = Test_Chn.input_DNN(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, DNN_time = Test_DNN.Unsupervised_Timeslot_Allocation_test(input_desired_signal, input_inter_signal, input_traffic_demand)

            DNN_timeslot = np.trunc(DNN_timeslot)

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, equal_power, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)

            print(DNN_timeslot)

            demand_gap = np.mean(np.sum(np.abs(test_traffic_demand - prop_data_rate), (1, 2)))
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))

            print('###############################################################')
            print('Unsupervised Learning for Timeslot Allocation (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Demand gap: \n' + str(demand_gap) + '[Mbps]')  # Revision
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(DNN_time * 1000.0) + '[msec]')
            print('###############################################################')

        if self.DL_scheme_val == 2:
            if self.keep_train == 1:
                print('###############################################################')
                print('DQN-based Timeslot Allocation Training Process (Only timeslot)')
                print('Making channel sample : %d' % self.num_train_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)

                train_channel_gain, train_group_channel_gain = Train_Chn.Channel_gain()
                train_traffic_demand = Train_Chn.Demand_generation()
                train_snapshot_index, train_group_snapshot_index = Train_Chn.snapshot_strategy1()

                Train_DNN = BH_dnn.Beamhopping_train(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                     self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt, self.reuse)

                input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Train_Chn.input_DNN3(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)
                Train_DNN.Single_DQN_Timeslot_Allocation_train(input_desired_signal, input_inter_signal, input_traffic_demand, self.epsilon, self.eps_decay, self.eps_min, self.discount_factor, input_channel, input_traffic_rate, input_value)


            print('###############################################################')
            print('DQN-based Timeslot Allocation Testing Process (Only timeslot)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt)

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            DNN_timeslot, DNN_time = Test_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                  input_channel, input_traffic_rate, input_value)
            print(DNN_timeslot)
            #for i in range(self.num_test_data_prop):
            #    print(DNN_timeslot[i])
            #equal_power = 10.0 ** (self.maximum_transmit_power / 10.0) / self.total_user / (self.num_timeslot / self.num_group)
            equal_power = np.ones([self.num_test_data_prop,self.num_fixed_beam,self.num_user])
            #equal_power, _ = Test_Chn.PA_without_US_entire_sample(DNN_timeslot, test_traffic_demand, test_snapshot_index, test_group_snapshot_index, test_group_channel_gain)
            '''
            for i in range(self.num_test_data_prop):
                #print(i)
                data, _ = Test_Chn.PA_without_US(DNN_timeslot[i], test_traffic_demand[i], test_snapshot_index[i], test_group_snapshot_index[i], test_group_channel_gain[i])
                equal_power[i] = data
            '''

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, equal_power, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, prop_power_allocation, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            print('###############################################################')
            print('DQN-based (only) Timeslot Allocation (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(DNN_time * 1000.0) + '[msec]')
            print('###############################################################')

        if self.DL_scheme_val == 3:
            if self.keep_train == 1:
                print('###############################################################')
                print('DQN-based Timeslot Allocation Training Process (Graph)')
                print('Making channel sample : %d' % self.num_train_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)

                train_channel_gain, train_group_channel_gain = Train_Chn.Channel_gain()
                train_traffic_demand = Train_Chn.Demand_generation()
                train_snapshot_index, train_group_snapshot_index = Train_Chn.snapshot_strategy1()

                Train_DNN = BH_dnn.Beamhopping_train(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                     self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt, self.reuse)

                input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Train_Chn.input_DNN3(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)

                Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_test_instances, self.test_seed)

                test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
                test_traffic_demand = Test_Chn.Demand_generation()
                test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

                test_input_desired_signal, test_input_inter_signal, test_input_traffic_demand, test_input_channel, test_input_traffic_rate, test_input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

                Train_DNN.Single_DQN_Timeslot_Allocation_graph(input_desired_signal, input_inter_signal, input_traffic_demand, self.epsilon, self.eps_decay, self.eps_min, self.discount_factor, input_channel, input_traffic_rate, input_value,
                                                               test_input_channel, test_input_traffic_rate, test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand, self.num_test_data_prop)

            print('###############################################################')
            print('DQN-based Timeslot Allocation Testing Process (Only timeslot)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 2, self.ckpt)

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            DNN_timeslot, DNN_time = Test_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                  input_channel, input_traffic_rate, input_value)

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, equal_power, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, prop_power_allocation, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            print('###############################################################')
            print('DQN-based (reward optimization) Timeslot Allocation (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(DNN_time * 1000.0) + '[msec]')
            print('###############################################################')

        if self.DL_scheme_val == 4:
            if self.keep_train == 1:
                print('###############################################################')
                print('Unsupervised Learning for Power Allocation Training Process')
                print('Making channel sample : %d' % self.num_train_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                '''
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)
                '''
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)
                train_channel_gain, train_group_channel_gain = Train_Chn.Channel_gain()
                train_traffic_demand = Train_Chn.Demand_generation()
                train_snapshot_index, train_group_snapshot_index = Train_Chn.snapshot_strategy1()

                Train_DNN = BH_dnn.Beamhopping_train(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                     self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt, self.reuse)
                input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Train_Chn.input_DNN3(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)

                Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                       self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                       self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 2, self.ckpt)
                DNN_timeslot, DNN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test(train_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, train_snapshot_index, train_group_snapshot_index,
                                                                                          input_channel, input_traffic_rate, input_value)
                DNN_input1, DNN_input2, DNN_input3 = Train_Chn.Normalized_input(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand, DNN_timeslot)
                minimum_power = Train_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
                minimum_power = np.reshape(minimum_power, [-1, self.total_user])
                #minimum_power = np.reshape(minimum_power * np.reshape(np.tile(np.expand_dims(DNN_timeslot, (2,3)), [1,1,self.num_active_beam,self.num_user]), [-1,self.num_fixed_beam,self.num_user]), [-1, self.total_user])

                Train_DNN.Unsupervised_Power_Allocation_train(input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, DNN_timeslot, DNN_input3)

            print('###############################################################')
            print('Unsupervised Learning for Power Allocation Testing Process')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)

            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)
            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 2, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, DNN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                      input_channel, input_traffic_rate, input_value)

            DNN_input1, DNN_input2, DNN_input3 = Test_Chn.Normalized_input(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand, DNN_timeslot)
            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt)

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            # minimum_input = np.reshape(minimum_power * np.reshape(np.tile(np.expand_dims(DNN_timeslot, (2,3)), [1,1,self.num_active_beam,self.num_user]), [-1,self.num_fixed_beam,self.num_user]), [-1, self.total_user])
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])
            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            minimum_outage_rate = Test_Chn.outage_rate(minimum_capacity, test_traffic_demand)
            minimum_energy_efficiency = Test_Chn.energy_efficiency(minimum_capacity, minimum_power, DNN_timeslot, test_group_snapshot_index)
            minimum_total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            minimum_total_power = np.mean(np.sum(minimum_total_power, (1, 2, 3)))
            minimum_capacity = np.mean(np.sum(minimum_capacity, (1,2)))
            minimum_EE = np.mean(minimum_energy_efficiency)

            DNN_power, DNN_time = Test_DNN.Unsupervised_Power_Allocation_test(DNN_timeslot, DNN_input3, minimum_input)
            power_alloc = np.expand_dims(DNN_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, DNN_power, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(DNN_power, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            print('###############################################################')
            print('Unsupervised Learning for Power Allocation (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('\n')
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Minimum capacity: \n' + str(minimum_capacity) + '[Mbps]')
            print('Minimum energy efficiency: \n' + str(minimum_EE) + '[Mbps/J]')
            print('Minimum outage rate: \n' + str(minimum_outage_rate) + '%')
            print('Minimum power consumption: \n' + str(10.0 * np.log10(minimum_total_power)) + '[dBm]')
            print('\n')
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(DNN_time * 1000.0) + '[msec]')
            print('###############################################################')

        if self.DL_scheme_val == 5:
            if self.keep_train == 1:
                print('###############################################################')
                print('Unsupervised Learning (Capacity Estimation) for Power Allocation Training Process')
                print('Making channel sample : %d' % self.num_train_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)
                train_channel_gain, train_group_channel_gain = Train_Chn.Channel_gain()
                train_traffic_demand = Train_Chn.Demand_generation()
                train_snapshot_index, train_group_snapshot_index = Train_Chn.snapshot_strategy1()

                Train_DNN = BH_dnn.Beamhopping_train(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                     self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt, self.reuse)
                input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Train_Chn.input_DNN3(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)

                Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                       self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                       self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 6, self.ckpt)
                DNN_timeslot, DNN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test2(train_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, train_snapshot_index, train_group_snapshot_index,
                                                                                          input_channel, input_traffic_rate, input_value)
                DNN_input1, DNN_input2, DNN_input3 = Train_Chn.Normalized_input(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand, DNN_timeslot)
                minimum_power = Train_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
                minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
                minimum_capacity = Train_Chn.data_rate_proposed(train_group_channel_gain, minimum_power_alloc, train_snapshot_index, train_group_snapshot_index, DNN_timeslot)
                minimum_power = np.reshape(minimum_power, [-1, self.total_user])
                capacity_min, capacity_max = Train_Chn.capacity_min_max(input_traffic_rate, input_channel, DNN_timeslot, minimum_capacity)
                #capacity_min = np.reshape(np.sum(capacity_min, 3), [-1,self.total_user])
                #capacity_max = np.reshape(np.sum(capacity_max, 3), [-1,self.total_user])

                Train_DNN.Unsupervised_Capacity_Estimation_train(input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, DNN_timeslot, DNN_input3, capacity_min, capacity_max)

            print('###############################################################')
            print('Unsupervised Learning (Capacity Estimation) for Power Allocation Testing Process')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)
            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 6, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, DNN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test2(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                      input_channel, input_traffic_rate, input_value)

            DNN_input1, DNN_input2, DNN_input3 = Test_Chn.Normalized_input(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand, DNN_timeslot)
            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt)

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            minimum_outage_rate = Test_Chn.outage_rate(minimum_capacity, test_traffic_demand)
            minimum_energy_efficiency = Test_Chn.energy_efficiency(minimum_capacity, minimum_power, DNN_timeslot, test_group_snapshot_index)
            minimum_total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, DNN_timeslot, minimum_capacity)
            DNN_capacity, DNN_time = Test_DNN.Unsupervised_Capacity_Estimation_test(DNN_timeslot, DNN_input3, capacity_min, capacity_max)
            DNN_capacity = np.reshape(DNN_capacity, [-1, self.total_user]) * self.bandwidth
            input_traffic_rate = np.swapaxes(tf.reshape(DNN_capacity / self.bandwidth, [-1, self.num_group, self.num_active_beam, self.num_user]), 2, 3)
            input_traffic_rate = np.expand_dims(input_traffic_rate, 3) * np.eye(self.num_active_beam)

            DNN_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            DNN_power = np.reshape(DNN_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(DNN_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, DNN_power, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(DNN_power, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            minimum_total_power = np.mean(np.sum(minimum_total_power, (1, 2, 3)))
            minimum_capacity = np.mean(np.sum(minimum_capacity, (1,2)))
            minimum_EE = np.mean(minimum_energy_efficiency)

            print('###############################################################')
            print('Unsupervised Learning (Capacity Estimation) for Power Allocation (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('\n')
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Minimum capacity: \n' + str(minimum_capacity) + '[Mbps]')
            print('Minimum energy efficiency: \n' + str(minimum_EE) + '[Mbps/J]')
            print('Minimum outage rate: \n' + str(minimum_outage_rate) + '%')
            print('Minimum power consumption: \n' + str(10.0 * np.log10(minimum_total_power)) + '[dBm]')
            print('\n')
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(DNN_time * 1000.0) + '[msec]')
            print('###############################################################')

        if self.DL_scheme_val == 6:
            if self.keep_train == 1:
                print('###############################################################')
                print('DQN-based Timeslot Allocation Training Process (Reward Optimization)')
                print('Making channel sample : %d' % self.num_train_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Train_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                         self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                         self.num_train_instances, self.train_seed)

                train_channel_gain, train_group_channel_gain = Train_Chn.Channel_gain()
                train_traffic_demand = Train_Chn.Demand_generation()
                train_snapshot_index, train_group_snapshot_index = Train_Chn.snapshot_strategy1()

                Train_DNN = BH_dnn.Beamhopping_train(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                     self.num_train_instances, self.train_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt, self.reuse)

                input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Train_Chn.input_DNN3(train_group_channel_gain, train_snapshot_index, train_group_snapshot_index, train_traffic_demand)
                #Train_DNN.Reward_Optimized_DQN_Timeslot_Allocation_train(input_desired_signal, input_inter_signal, input_traffic_demand, self.epsilon, self.eps_decay, self.eps_min, self.discount_factor, input_channel, input_traffic_rate, input_value)
                Train_DNN.Single_DQN_Timeslot_Allocation_train2(input_desired_signal, input_inter_signal, input_traffic_demand, self.epsilon, self.eps_decay, self.eps_min, self.discount_factor, input_channel, input_traffic_rate, input_value)

            print('###############################################################')
            print('DQN-based Timeslot Allocation Testing Process (Reward Optimization)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, self.DL_scheme_val, self.ckpt)

            input_desired_signal, input_inter_signal, input_traffic_demand = Test_Chn.input_DNN(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            DNN_timeslot, DNN_time = Test_DNN.Single_DQN_Timeslot_Allocation_test2(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                            input_channel, input_traffic_rate, input_value)
            print(DNN_timeslot)
            #for i in range(self.num_test_data_prop):
            #    print(DNN_timeslot[i])
            #equal_power = 10.0 ** (self.maximum_transmit_power / 10.0) / self.total_user / (self.num_timeslot / self.num_group)
            equal_power = np.ones([self.num_test_data_prop,self.num_fixed_beam,self.num_user])
            #equal_power, _ = Test_Chn.PA_without_US_entire_sample(DNN_timeslot, test_traffic_demand, test_snapshot_index, test_group_snapshot_index, test_group_channel_gain)
            '''
            for i in range(self.num_test_data_prop):
                #print(i)
                data, _ = Test_Chn.PA_without_US(DNN_timeslot[i], test_traffic_demand[i], test_snapshot_index[i], test_group_snapshot_index[i], test_group_channel_gain[i])
                equal_power[i] = data
            '''

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, equal_power, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, prop_power_allocation, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            print('###############################################################')
            print('DQN-based (reward optimization) Timeslot Allocation (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(DNN_time * 1000.0) + '[msec]')
            print('###############################################################')

        return 0

    def heuristic_result(self):
        # test step
        if self.Heuristic_scheme_val == 0:
            pass

        if self.Heuristic_scheme_val == 1:
            if self.keep_train == 1:
                print('###############################################################')
                print('Proposed Heuristic Algorithm (Generation)')
                print('Making channel sample : %d' % self.num_test_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                        self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                        self.num_test_instances, self.test_seed)

                test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
                test_traffic_demand = Test_Chn.Demand_generation()
                test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

                start_time = time_module.time()
                prop_power_allocation, prop_time_allocation = Test_Chn.Proposed_Heuristic_Algorithm(test_channel_gain, test_traffic_demand, test_snapshot_index, test_group_snapshot_index)
                prop_time = time_module.time() - start_time

                np.savetxt(self.savefile_prop_heuristic[0] + '.csv', np.reshape(prop_power_allocation, [-1, self.num_fixed_beam*self.num_user]), delimiter=",")
                np.savetxt(self.savefile_prop_heuristic[1] + '.csv', np.reshape(prop_time_allocation, [-1, self.num_group]), delimiter=",")
                np.savetxt(self.savefile_prop_heuristic[2] + '.csv', [prop_time], delimiter=",")

            if self.keep_train == 0:
                print('###############################################################')
                print('Proposed Heuristic Algorithm (Loading)')
                print('Making channel sample : %d' % self.num_test_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                        self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                        self.num_test_instances, self.test_seed)

                test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
                test_traffic_demand = Test_Chn.Demand_generation()
                test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

                prop_power_allocation = np.loadtxt(self.savefile_prop_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
                prop_time_allocation = np.loadtxt(self.savefile_prop_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)
                prop_time = np.loadtxt(self.savefile_prop_heuristic[2] + '.csv', delimiter=",", dtype=np.float32)

            print(prop_time_allocation)

            prop_power_allocation = np.reshape(prop_power_allocation, [self.num_test_data_prop, self.num_fixed_beam, self.num_user])
            prop_time_allocation = np.reshape(prop_time_allocation, [self.num_test_data_prop, self.num_group])
            '''
            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, prop_time_allocation)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))
            '''
            group_power_val = np.tile(np.expand_dims(prop_power_allocation, (1,2)), [1, self.num_group, self.num_fixed_beam, 1, 1])
            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, group_power_val, test_snapshot_index, test_group_snapshot_index, prop_time_allocation)

            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, prop_power_allocation, prop_time_allocation, test_group_snapshot_index)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(prop_time_allocation, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            print('###############################################################')
            print('Proposed Heuristic Algorithm (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(np.sum(prop_time) * 1000.0) + '[msec]')
            print('###############################################################')

        if self.Heuristic_scheme_val == 2:
            if self.keep_train == 1:
                print('###############################################################')
                print('Conventional Per-slot Heuristic Algorithm (Generation)')
                print('Making channel sample : %d' % self.num_test_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                        self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                        self.num_test_instances, self.test_seed)

                test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
                test_traffic_demand = Test_Chn.Demand_generation()
                test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

                start_time = time_module.time()
                conv_power_allocation, conv_time_allocation = Test_Chn.Conventional_Heuristic_Algorithm(test_channel_gain, test_traffic_demand, test_snapshot_index, test_group_snapshot_index)
                conv_time = time_module.time() - start_time

                np.savetxt(self.savefile_conv_heuristic[0] + '.csv', np.reshape(conv_time_allocation, [-1, self.num_group]), delimiter=",")
                np.savetxt(self.savefile_conv_heuristic[1] + '.csv', [conv_time], delimiter=",")

            if self.keep_train == 0:
                print('###############################################################')
                print('Conventional Per-slot Heuristic Algorithm (Loading)')
                print('Making channel sample : %d' % self.num_test_instances)
                print('The number of fixed beam : %d' % self.num_fixed_beam)
                print('The number of active beam : %d' % self.num_active_beam)
                print('The number of users : %d' % self.num_user)
                print('Average traffic demand : %d [Mbps]' % self.average_demand)
                print('###############################################################')
                time_module.sleep(2)
                Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                        self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                        self.num_test_instances, self.test_seed)

                test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
                test_traffic_demand = Test_Chn.Demand_generation()
                test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

                conv_time_allocation = np.loadtxt(self.savefile_conv_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
                conv_time = np.loadtxt(self.savefile_conv_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)

            conv_time_allocation = np.reshape(conv_time_allocation, [self.num_test_data_prop, self.num_timeslot, self.num_group])
            timeslot_data = np.sum(conv_time_allocation, 1)
            timeslot_data = np.where(timeslot_data == 0, 1, timeslot_data)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            power_alloc = np.expand_dims(minimum_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, minimum_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            print('###############################################################')
            print('Conventional Per-slot Heuristic Algorithm (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(np.sum(conv_time) * 1000.0) + '[msec]')
            print('###############################################################')

        if self.Heuristic_scheme_val == 3:
            print('###############################################################')
            print('Trust region-based IPM algorithm (per-slot) (Generation)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            # generation timeslot data (per-slot algorithm)
            ########################################################################################################################################
            savefile_str = ('Altitude' + str(self.Altitude) + 'active_beam' + str(self.num_active_beam) +
                            'fixed_beam' + str(self.num_fixed_beam) + 'user' + str(self.num_user) + 'timeslot' + str(self.num_timeslot) +
                            'elevation_angle' + str(self.num_angle) + 'Tx_power' + str(self.maximum_transmit_power) +
                            'average_demand' + str(self.average_demand) + 'bandwidth' + str(self.bandwidth)
                            )
            self.savefile_conv_heuristic = [
                './Per_slot/time_allocation_val/' + savefile_str,
                './Per_slot/computation_time/' + savefile_str
            ]

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            conv_time_allocation = np.loadtxt(self.savefile_conv_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
            conv_time = np.loadtxt(self.savefile_conv_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)

            conv_time_allocation = np.reshape(conv_time_allocation, [self.num_test_data_prop, self.num_timeslot, self.num_group])
            timeslot_data = np.sum(conv_time_allocation, 1)
            timeslot_data = np.where(timeslot_data == 0, 1, timeslot_data)
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_perslot(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            timeslot_alloc = np.tile(np.reshape(timeslot_data, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])

            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            minimum_outage_rate = Test_Chn.outage_rate(minimum_capacity, test_traffic_demand)
            minimum_energy_efficiency = Test_Chn.energy_efficiency(minimum_capacity, minimum_power, timeslot_data, test_group_snapshot_index)
            minimum_total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)

            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            minimum_total_power = np.mean(np.sum(minimum_total_power, (1, 2, 3)))
            minimum_capacity = np.mean(np.sum(minimum_capacity, (1, 2)))
            minimum_EE = np.mean(minimum_energy_efficiency)

            print('###############################################################')
            print('Trust region-based IPM algorithm (per-slot) (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('\n')
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Minimum capacity: \n' + str(minimum_capacity) + '[Mbps]')
            print('Minimum energy efficiency: \n' + str(minimum_EE) + '[Mbps/J]')
            print('Minimum outage rate: \n' + str(minimum_outage_rate) + '%')
            print('Minimum power consumption: \n' + str(10.0 * np.log10(minimum_total_power)) + '[dBm]')
            print('\n')
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(np.sum(IPM_time) * 1000.0) + '[msec]')
            print('###############################################################')

        if self.Heuristic_scheme_val == 4:
            print('###############################################################')
            print('Trust region-based IPM algorithm (iterative) (Generation)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            # generation timeslot data (Iterative algorithm)
            ########################################################################################################################################
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            prop_power_allocation = np.loadtxt(self.savefile_prop_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
            prop_time_allocation = np.loadtxt(self.savefile_prop_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)
            prop_time = np.loadtxt(self.savefile_prop_heuristic[2] + '.csv', delimiter=",", dtype=np.float32)

            prop_power_allocation = np.reshape(prop_power_allocation, [self.num_test_data_prop, self.num_fixed_beam, self.num_user])
            prop_time_allocation = np.reshape(prop_time_allocation, [self.num_test_data_prop, self.num_group])
            timeslot_data = prop_time_allocation
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_iterative(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            timeslot_alloc = np.tile(np.reshape(timeslot_data, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])
            desired_SNR = input_desired_signal * IPM_power
            inter_SNR = np.sum(input_inter_signal * np.expand_dims(IPM_power, 1), -1) + 1.0
            prop_data_rate2 = self.bandwidth / self.num_user * timeslot_alloc * np.log2(1.0 + desired_SNR / inter_SNR)
            prop_data_rate2 = np.reshape(prop_data_rate2, [-1, self.num_fixed_beam, self.num_user])

            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            minimum_outage_rate = Test_Chn.outage_rate(minimum_capacity, test_traffic_demand)
            minimum_energy_efficiency = Test_Chn.energy_efficiency(minimum_capacity, minimum_power, timeslot_data, test_group_snapshot_index)
            minimum_total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            test = prop_data_rate >= test_traffic_demand * 0.999
            #print(np.sum(test, (1,2)))
            test2 = minimum_capacity >= test_traffic_demand * 0.999
            #print(np.sum(test2, (1,2)))

            test3 = np.where(((np.sum(test2, (1,2)) == self.total_user) * 1.0 - (np.sum(test, (1,2)) < self.total_user) * 1.0) == 0, 1, 0)

            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            minimum_total_power = np.mean(np.sum(minimum_total_power, (1, 2, 3)))
            minimum_capacity = np.mean(np.sum(minimum_capacity, (1, 2)))
            minimum_EE = np.mean(minimum_energy_efficiency)

            print('###############################################################')
            print('Trust region-based IPM algorithm (iterative) (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('\n')
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Minimum capacity: \n' + str(minimum_capacity) + '[Mbps]')
            print('Minimum energy efficiency: \n' + str(minimum_EE) + '[Mbps/J]')
            print('Minimum outage rate: \n' + str(minimum_outage_rate) + '%')
            print('Minimum power consumption: \n' + str(10.0 * np.log10(minimum_total_power)) + '[dBm]')
            print('\n')
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(np.sum(IPM_time) * 1000.0) + '[msec]')
            print('###############################################################')

        if self.Heuristic_scheme_val == 5:
            print('###############################################################')
            print('Trust region-based IPM algorithm (DQN) (Generation)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            #generation timeslot data (Proposed DNN-based timeslot allocation)
            ########################################################################################################################################
            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                   self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                   self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 2, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, _ = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                               input_channel, input_traffic_rate, input_value)
            timeslot_data = DNN_timeslot
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            cuda.select_device(0)
            cuda.close()

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_DQN(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            timeslot_alloc = np.tile(np.reshape(timeslot_data, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])
            desired_SNR = input_desired_signal * IPM_power
            inter_SNR = np.sum(input_inter_signal * np.expand_dims(IPM_power, 1), -1) + 1.0
            prop_data_rate2 = self.bandwidth / self.num_user * timeslot_alloc * np.log2(1.0 + desired_SNR / inter_SNR)
            prop_data_rate2 = np.reshape(prop_data_rate2, [-1, self.num_fixed_beam, self.num_user])

            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            minimum_outage_rate = Test_Chn.outage_rate(minimum_capacity, test_traffic_demand)
            minimum_energy_efficiency = Test_Chn.energy_efficiency(minimum_capacity, minimum_power, timeslot_data, test_group_snapshot_index)
            minimum_total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            test = prop_data_rate >= test_traffic_demand * 0.999
            #print(np.sum(test, (1,2)))
            test2 = minimum_capacity >= test_traffic_demand * 0.999
            #print(np.sum(test2, (1,2)))

            test3 = np.where(((np.sum(test2, (1,2)) == self.total_user) * 1.0 - (np.sum(test, (1,2)) < self.total_user) * 1.0) == 0, 1, 0)

            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            minimum_total_power = np.mean(np.sum(minimum_total_power, (1, 2, 3)))
            minimum_capacity = np.mean(np.sum(minimum_capacity, (1, 2)))
            minimum_EE = np.mean(minimum_energy_efficiency)

            print('###############################################################')
            print('Trust region-based IPM algorithm (DQN) (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('\n')
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Minimum capacity: \n' + str(minimum_capacity) + '[Mbps]')
            print('Minimum energy efficiency: \n' + str(minimum_EE) + '[Mbps/J]')
            print('Minimum outage rate: \n' + str(minimum_outage_rate) + '%')
            print('Minimum power consumption: \n' + str(10.0 * np.log10(minimum_total_power)) + '[dBm]')
            print('\n')
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(np.sum(IPM_time) * 1000.0) + '[msec]')
            print('###############################################################')

        if self.Heuristic_scheme_val == 6:
            print('###############################################################')
            print('Trust region-based IPM algorithm (Dueling DQN) (Generation)')
            print('Making channel sample : %d' % self.num_test_instances)
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('###############################################################')
            time_module.sleep(2)
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            #generation timeslot data (Proposed DNN-based timeslot allocation)
            ########################################################################################################################################
            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                   self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                   self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 6, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, _ = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test2(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                               input_channel, input_traffic_rate, input_value)
            timeslot_data = DNN_timeslot
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            cuda.select_device(0)
            cuda.close()

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_Dueling_DQN(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            timeslot_alloc = np.tile(np.reshape(timeslot_data, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])
            desired_SNR = input_desired_signal * IPM_power
            inter_SNR = np.sum(input_inter_signal * np.expand_dims(IPM_power, 1), -1) + 1.0
            prop_data_rate2 = self.bandwidth / self.num_user * timeslot_alloc * np.log2(1.0 + desired_SNR / inter_SNR)
            prop_data_rate2 = np.reshape(prop_data_rate2, [-1, self.num_fixed_beam, self.num_user])

            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            minimum_outage_rate = Test_Chn.outage_rate(minimum_capacity, test_traffic_demand)
            minimum_energy_efficiency = Test_Chn.energy_efficiency(minimum_capacity, minimum_power, timeslot_data, test_group_snapshot_index)
            minimum_total_power = np.expand_dims(minimum_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            test = prop_data_rate >= test_traffic_demand * 0.999
            #print(np.sum(test, (1,2)))
            test2 = minimum_capacity >= test_traffic_demand * 0.999
            #print(np.sum(test2, (1,2)))

            test3 = np.where(((np.sum(test2, (1,2)) == self.total_user) * 1.0 - (np.sum(test, (1,2)) < self.total_user) * 1.0) == 0, 1, 0)

            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            minimum_total_power = np.mean(np.sum(minimum_total_power, (1, 2, 3)))
            minimum_capacity = np.mean(np.sum(minimum_capacity, (1, 2)))
            minimum_EE = np.mean(minimum_energy_efficiency)

            print('###############################################################')
            print('Trust region-based IPM algorithm (Dueling DQN) (Performance)')
            print('The number of fixed beam : %d' % self.num_fixed_beam)
            print('The number of active beam : %d' % self.num_active_beam)
            print('The number of users : %d' % self.num_user)
            print('\n')
            print('Average traffic demand : %d [Mbps]' % self.average_demand)
            print('Minimum capacity: \n' + str(minimum_capacity) + '[Mbps]')
            print('Minimum energy efficiency: \n' + str(minimum_EE) + '[Mbps/J]')
            print('Minimum outage rate: \n' + str(minimum_outage_rate) + '%')
            print('Minimum power consumption: \n' + str(10.0 * np.log10(minimum_total_power)) + '[dBm]')
            print('\n')
            print('Capacity: \n' + str(capacity) + '[Mbps]')
            print('Energy efficiency: \n' + str(EE) + '[Mbps/J]')
            print('Outage rate: \n' + str(outage_rate) + '%')
            print('Power consumption: \n' + str(10.0 * np.log10(total_power)) + '[dBm]')
            print('Computational time: \n' + str(np.sum(IPM_time) * 1000.0) + '[msec]')
            print('###############################################################')

    def graph_result(self, heuristic_scheme_val=0, DL_scheme_val=0, rho_val=0.99):
        # for graph
        power_val = 0
        demand_gap = 0
        reward_val = 0
        comp_time = 0
        if heuristic_scheme_val == 0:
            pass
        if heuristic_scheme_val == 1:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            prop_power_allocation = np.loadtxt(self.savefile_prop_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
            prop_time_allocation = np.loadtxt(self.savefile_prop_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)
            prop_time = np.loadtxt(self.savefile_prop_heuristic[2] + '.csv', delimiter=",", dtype=np.float32)

            prop_power_allocation = np.reshape(prop_power_allocation, [self.num_test_data_prop, self.num_fixed_beam, self.num_user])
            prop_time_allocation = np.reshape(prop_time_allocation, [self.num_test_data_prop, self.num_group])

            group_power_val = np.tile(np.expand_dims(prop_power_allocation, (1, 2)), [1, self.num_group, self.num_fixed_beam, 1, 1])
            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, group_power_val, test_snapshot_index, test_group_snapshot_index, prop_time_allocation)

            demand_gap = np.abs(test_traffic_demand - prop_data_rate)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(prop_time_allocation, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            power_val = total_power

            comp_time = prop_time

            timeslot_for_state = np.tile(np.reshape(prop_time_allocation, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_for_state = np.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
            equal_power_val = 10.0 ** (self.maximum_transmit_power / 10.0) * np.ones(self.num_fixed_beam * self.num_user) / self.num_fixed_beam / self.num_user / timeslot_for_state
            equal_power_val = np.tile(np.expand_dims(np.reshape(equal_power_val, [-1,self.num_fixed_beam,self.num_user]), (1,2)), [1,self.num_group,self.num_fixed_beam,1,1])
            reward_val = 0

        if heuristic_scheme_val == 2:
            pass

        if heuristic_scheme_val == 3:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            #generation timeslot data (per-slot algorithm)
            ########################################################################################################################################
            savefile_str = ('Altitude' + str(self.Altitude) + 'active_beam' + str(self.num_active_beam) +
                            'fixed_beam' + str(self.num_fixed_beam) + 'user' + str(self.num_user) + 'timeslot' + str(self.num_timeslot) +
                            'elevation_angle' + str(self.num_angle) + 'Tx_power' + str(self.maximum_transmit_power) +
                            'average_demand' + str(self.average_demand) + 'bandwidth' + str(self.bandwidth)
                            )
            self.savefile_conv_heuristic = [
                './Per_slot/time_allocation_val/' + savefile_str,
                './Per_slot/computation_time/' + savefile_str
            ]
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            conv_time_allocation = np.loadtxt(self.savefile_conv_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
            conv_time = np.loadtxt(self.savefile_conv_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)

            conv_time_allocation = np.reshape(conv_time_allocation, [self.num_test_data_prop, self.num_timeslot, self.num_group])
            timeslot_data = np.sum(conv_time_allocation, 1)
            timeslot_data = np.where(timeslot_data == 0, 1, timeslot_data)
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_perslot(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            power_val = total_power
            demand_gap = EE
            reward_val = outage_rate
            comp_time = np.sum(IPM_time) + conv_time

        if heuristic_scheme_val == 4:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            #generation timeslot data (Iterative algorithm)
            ########################################################################################################################################
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            prop_power_allocation = np.loadtxt(self.savefile_prop_heuristic[0] + '.csv', delimiter=",", dtype=np.float32)
            prop_time_allocation = np.loadtxt(self.savefile_prop_heuristic[1] + '.csv', delimiter=",", dtype=np.float32)
            prop_time = np.loadtxt(self.savefile_prop_heuristic[2] + '.csv', delimiter=",", dtype=np.float32)

            prop_power_allocation = np.reshape(prop_power_allocation, [self.num_test_data_prop, self.num_fixed_beam, self.num_user])
            prop_time_allocation = np.reshape(prop_time_allocation, [self.num_test_data_prop, self.num_group])
            timeslot_data = prop_time_allocation
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_iterative(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            power_val = total_power
            demand_gap = EE
            reward_val = outage_rate
            comp_time = np.sum(IPM_time) + prop_time

        if heuristic_scheme_val == 5:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            #generation timeslot data (Proposed DNN-based timeslot allocation)
            ########################################################################################################################################
            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                   self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                   self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 2, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, DNN_execute_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                               input_channel, input_traffic_rate, input_value)
            timeslot_data = DNN_timeslot
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_DQN(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            power_val = total_power
            demand_gap = EE
            reward_val = outage_rate
            comp_time = np.sum(IPM_time) + DNN_execute_time

        if heuristic_scheme_val == 6:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            #generation timeslot data (Proposed DNN-based timeslot allocation)
            ########################################################################################################################################
            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                   self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                   self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 6, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, DNN_execute_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test2(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                               input_channel, input_traffic_rate, input_value)
            timeslot_data = DNN_timeslot
            ########################################################################################################################################

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, timeslot_data)
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, timeslot_data, minimum_capacity)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            IPM_power, IPM_time = Test_Chn.TR_IPM_PA_algorithm_Dueling_DQN(input_desired_signal, input_inter_signal, capacity_min, input_channel, input_traffic_demand, timeslot_data, minimum_input, self.keep_train)
            IPM_power = np.reshape(IPM_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(IPM_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, timeslot_data)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, IPM_power, timeslot_data, test_group_snapshot_index)
            total_power = np.expand_dims(IPM_power, 1) * np.expand_dims(timeslot_data, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            timeslot_for_state = np.tile(np.reshape(DNN_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_for_state = np.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
            equal_power_val = 10.0 ** (self.maximum_transmit_power / 10.0) * np.ones(self.num_fixed_beam * self.num_user) / self.num_fixed_beam / self.num_user / timeslot_for_state
            equal_power_val = np.tile(np.expand_dims(np.reshape(equal_power_val, [-1, self.num_fixed_beam, self.num_user]), (1, 2)), [1, self.num_group, self.num_fixed_beam, 1, 1])
            power_val = reward_val

            # power_val = total_power
            demand_gap = EE
            reward_val = outage_rate
            comp_time = np.sum(IPM_time) + DNN_execute_time

        if DL_scheme_val == 0:
            pass

        if DL_scheme_val == 1:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, DL_scheme_val, self.ckpt)

            input_desired_signal, input_inter_signal, input_traffic_demand = Test_Chn.input_DNN(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            DNN_timeslot, DNN_time = Test_DNN.Unsupervised_Timeslot_Allocation_test(input_desired_signal, input_inter_signal, input_traffic_demand)

            start_time = time_module.time()
            DNN_timeslot = np.trunc(DNN_timeslot)

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, equal_power, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)

            demand_gap = np.abs(test_traffic_demand - prop_data_rate)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            end_time = time_module.time() - start_time

            power_val = total_power
            comp_time = DNN_time + end_time

            timeslot_for_state = np.tile(np.reshape(DNN_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_for_state = np.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
            equal_power_val = 10.0 ** (self.maximum_transmit_power / 10.0) * np.ones(self.num_fixed_beam * self.num_user) / self.num_fixed_beam / self.num_user / timeslot_for_state
            equal_power_val = np.tile(np.expand_dims(np.reshape(equal_power_val, [-1,self.num_fixed_beam,self.num_user]), (1,2)), [1,self.num_group,self.num_fixed_beam,1,1])
            reward_val = 0

        if DL_scheme_val == 2:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)

            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, DL_scheme_val, self.ckpt)

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            DNN_timeslot, DNN_time = Test_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                  input_channel, input_traffic_rate, input_value)

            start_time = time_module.time()

            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)

            prop_power_allocation = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            equal_power = np.expand_dims(prop_power_allocation, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, equal_power, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)

            demand_gap = np.abs(test_traffic_demand - prop_data_rate)
            total_power = np.expand_dims(prop_power_allocation, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot

            end_time = time_module.time() - start_time

            power_val = total_power
            comp_time = DNN_time + end_time

            timeslot_for_state = np.tile(np.reshape(DNN_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_for_state = np.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
            equal_power_val = 10.0 ** (self.maximum_transmit_power / 10.0) * np.ones(self.num_fixed_beam * self.num_user) / self.num_fixed_beam / self.num_user / timeslot_for_state
            equal_power_val = np.tile(np.expand_dims(np.reshape(equal_power_val, [-1,self.num_fixed_beam,self.num_user]), (1,2)), [1,self.num_group,self.num_fixed_beam,1,1])
            reward_val = 0

        if DL_scheme_val == 3:
            demand_gap = np.loadtxt(self.ckpt[DL_scheme_val] + '_EE.csv', delimiter=",", dtype=np.float32)
            power_val = np.loadtxt(self.ckpt[DL_scheme_val] + '_reward.csv', delimiter=",", dtype=np.float32)
            reward_val = np.loadtxt(self.ckpt[DL_scheme_val] + '_outage.csv', delimiter=",", dtype=np.float32)

            comp_time = 0.0

        if DL_scheme_val == 4:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)
            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 6, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            #DNN_timeslot, DQN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
            #                                                                          input_channel, input_traffic_rate, input_value)
            DNN_timeslot, DQN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test2(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                       input_channel, input_traffic_rate, input_value)

            DNN_input1, DNN_input2, DNN_input3 = Test_Chn.Normalized_input(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand, DNN_timeslot)
            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 4, self.ckpt)

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot)
            # minimum_input = np.reshape(minimum_power * np.reshape(np.tile(np.expand_dims(DNN_timeslot, (2,3)), [1,1,self.num_active_beam,self.num_user]), [-1,self.num_fixed_beam,self.num_user]), [-1, self.total_user])
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])

            DNN_power, DNN_time = Test_DNN.Unsupervised_Power_Allocation_test(DNN_timeslot, DNN_input3, minimum_input)
            #DNN_power = minimum_power
            power_alloc = np.expand_dims(DNN_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, DNN_power, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(DNN_power, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            power_val = total_power
            demand_gap = EE
            reward_val = outage_rate
            comp_time = DNN_time + DQN_time

        if DL_scheme_val == 5:
            Test_Chn = BH_channel.Channel_generator(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                                    self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                                    self.num_test_instances, self.test_seed)
            test_channel_gain, test_group_channel_gain = Test_Chn.Channel_gain()
            test_traffic_demand = Test_Chn.Demand_generation()
            test_snapshot_index, test_group_snapshot_index = Test_Chn.snapshot_strategy1()

            Timeslot_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 6, self.ckpt)
            input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value = Test_Chn.input_DNN3(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand)
            #DNN_timeslot, DQN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
            #                                                                          input_channel, input_traffic_rate, input_value)
            DNN_timeslot, DQN_time = Timeslot_DNN.Single_DQN_Timeslot_Allocation_test2(test_channel_gain, input_desired_signal, input_inter_signal, input_traffic_demand, test_snapshot_index, test_group_snapshot_index,
                                                                                      input_channel, input_traffic_rate, input_value)

            DNN_input1, DNN_input2, DNN_input3 = Test_Chn.Normalized_input(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_traffic_demand, DNN_timeslot)
            Test_DNN = BH_dnn.Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                               self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                               self.num_test_instances, self.test_seed, self.learning_rate, self.training_epochs, self.batch_size, self.lambda_val, 5, self.ckpt)

            minimum_power = Test_Chn.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, DNN_timeslot, rho_val=rho_val)
            minimum_input = np.reshape(minimum_power, [-1, self.total_user])
            minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
            minimum_capacity = Test_Chn.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)

            capacity_min, capacity_max = Test_Chn.capacity_min_max(input_traffic_rate, input_channel, DNN_timeslot, minimum_capacity, rho_val=rho_val)
            DNN_capacity, DNN_time = Test_DNN.Unsupervised_Capacity_Estimation_test(DNN_timeslot, DNN_input3, capacity_min, capacity_max)
            DNN_capacity = np.reshape(DNN_capacity, [-1, self.total_user]) * self.bandwidth
            input_traffic_rate = np.swapaxes(tf.reshape(DNN_capacity / self.bandwidth, [-1, self.num_group, self.num_active_beam, self.num_user]), 2, 3)
            input_traffic_rate = np.expand_dims(input_traffic_rate, 3) * np.eye(self.num_active_beam)

            start_time = time_module.time()
            DNN_power = Test_Chn.matrix_inversion_based_power_allocation2(input_traffic_rate, input_channel, DNN_timeslot)
            matrix_inversion_time = time_module.time() - start_time
            DNN_power = np.reshape(DNN_power, [-1, self.num_fixed_beam, self.num_user])
            power_alloc = np.expand_dims(DNN_power, (1, 2))

            prop_data_rate = Test_Chn.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, DNN_timeslot)
            outage_rate = Test_Chn.outage_rate(prop_data_rate, test_traffic_demand)
            energy_efficiency = Test_Chn.energy_efficiency(prop_data_rate, DNN_power, DNN_timeslot, test_group_snapshot_index)
            total_power = np.expand_dims(DNN_power, 1) * np.expand_dims(DNN_timeslot, (2, 3)) * np.expand_dims(test_group_snapshot_index, -1) / self.num_timeslot
            total_power = np.mean(np.sum(total_power, (1, 2, 3)))
            capacity = np.mean(np.sum(prop_data_rate, (1, 2)))
            EE = np.mean(energy_efficiency)

            power_val = total_power
            demand_gap = EE
            reward_val = outage_rate
            comp_time = DNN_time + DQN_time + matrix_inversion_time

        return power_val, demand_gap, reward_val, comp_time
