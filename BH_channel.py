import numpy as np
import time
import math
import random
import scipy as sp
from scipy.spatial.distance import cdist
from itertools import product

class Channel_generator:
    def __init__(self, Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, num_instances, seed):
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
        self.elevation_angle_candidate = elevation_angle_candidate
        self.R_min = R_min
        self.average_demand = average_demand

        self.num_instances = num_instances
        self.seed = seed

        self.num_data_conv = self.num_instances * self.num_timeslot * self.num_angle
        self.num_data_prop = self.num_instances * self.num_angle

        self.beam_radius = self.Altitude * np.tan(np.radians(self.dB3_angle))
        self.total_user = self.num_user * self.num_fixed_beam

    def Generation_beam_deployment(self):

        #WCL
        beam_deployment = [[0, 0],
                           [0, 3],
                           [-2*np.sqrt(3), 3],
                           [-2*np.sqrt(3), 0],
                           [np.sqrt(3), 0],
                           [np.sqrt(3)/2, 9/2],
                           [-3*np.sqrt(3)/2, 9/2],
                           [-np.sqrt(3)/2, 3/2],
                           [-np.sqrt(3)/2, -3/2],
                           [np.sqrt(3)/2, 3/2],
                           [-3*np.sqrt(3)/2, 3/2],
                           [3*np.sqrt(3)/2,-3/2],
                           [np.sqrt(3)/2, -3/2],
                           [np.sqrt(3), 3],
                           [-np.sqrt(3), 3],
                           [-np.sqrt(3), 0]]

        #scattered
        beam_deployment1 = np.array([[0,0],
                                    [-3,0],
                                    [0,2*np.sqrt(3)],
                                    [-3,2*np.sqrt(3)]])
        beam_deployment = np.concatenate((beam_deployment1, beam_deployment1 + np.array([3/2,np.sqrt(3)/2]), beam_deployment1 + np.array([0,np.sqrt(3)]), beam_deployment1 + np.array([3/2,3/2*np.sqrt(3)])), axis=0)

        '''
        #converged
        beam_deployment1 = np.array([[0, 0],
                                     [3 / 2, np.sqrt(3) / 2],
                                     [0, np.sqrt(3)],
                                     [3 / 2, 3 / 2 * np.sqrt(3)]])
        beam_deployment = np.concatenate((beam_deployment1, beam_deployment1 + np.array([-3, 0]), beam_deployment1 + np.array([0, 2 * np.sqrt(3)]), beam_deployment1 + np.array([-3, 2 * np.sqrt(3)])), axis=0)
        '''
        beam_deployment = np.array(beam_deployment, np.float32)
        beam_deployment = beam_deployment * self.beam_radius
        beam_deployment = beam_deployment[:self.num_fixed_beam,:]

        return beam_deployment

    def Deployment_generation(self):
        np.random.seed(self.seed)
        satellite_deployment = np.zeros([self.num_angle, 3])
        user_deployment = np.zeros([self.num_instances, self.num_fixed_beam, self.num_user, 2])
        beam_deployment = self.Generation_beam_deployment()
        beam_deployment_expand = np.reshape(beam_deployment, [1,self.num_fixed_beam,1,2])

        satellite_deploy_X = -np.cos(self.elevation_angle_candidate)*self.Altitude
        satellite_deploy_Y = np.zeros(self.num_angle)
        satellite_deploy_Z = np.sin(self.elevation_angle_candidate)*self.Altitude
        satellite_deployment[:,0] = satellite_deploy_X
        satellite_deployment[:,1] = satellite_deploy_Y
        satellite_deployment[:,2] = satellite_deploy_Z

        user_distance = self.beam_radius * np.random.uniform(0, 1, [self.num_instances, self.num_fixed_beam, self.num_user])
        user_angle = 2.0 * np.pi * np.random.uniform(0, 1, [self.num_instances, self.num_fixed_beam, self.num_user])

        user_deployment[:,:,:,0] = user_distance * np.cos(user_angle) + beam_deployment_expand[:,:,:,0]
        user_deployment[:,:,:,1] = user_distance * np.sin(user_angle) + beam_deployment_expand[:,:,:,1]


        satellite_deployment = np.expand_dims(satellite_deployment, 0)
        satellite_deployment = np.tile(satellite_deployment, [self.num_instances, 1, 1])
        satellite_deployment = np.reshape(satellite_deployment, [-1,3])
        satellite_deployment = np.array(satellite_deployment, np.float32)

        user_deployment = np.expand_dims(user_deployment, 1)
        user_deployment = np.tile(user_deployment, [1, self.num_angle, 1, 1, 1])
        user_deployment = np.reshape(user_deployment, [-1, self.num_fixed_beam, self.num_user, 2])
        user_deployment = np.array(user_deployment, np.float32)

        elevation_angle = np.reshape(self.elevation_angle_candidate, [self.num_angle, 1])
        elevation_angle = np.tile(np.expand_dims(elevation_angle, 0), [self.num_instances, 1, 1])
        elevation_angle = np.reshape(elevation_angle, [-1,1])

        return satellite_deployment, user_deployment, elevation_angle

    def Generation_channelloss_angle_beampattern(self):
        satellite_deployment, user_deployment, elevation_angle = self.Deployment_generation()
        user_deployment = np.reshape(user_deployment, [-1,self.total_user,2])
        beam_deployment = self.Generation_beam_deployment()

        satellite_distance = np.sqrt((self.Earth_radius * np.sin(elevation_angle)) ** 2 + self.Altitude ** 2 + 2 * self.Altitude * self.Earth_radius) - self.Earth_radius * np.sin(elevation_angle)
        user_distance = np.sqrt(user_deployment[:, :, 0] ** 2 + user_deployment[:, :, 1] ** 2)

        distance_satel_to_user = np.sqrt(satellite_distance ** 2 + user_distance ** 2 - 2*satellite_distance*user_distance*np.cos(elevation_angle))

        PL_dB = 32.45 + 20*np.log10(self.carrier_frequency) + 20*np.log10(distance_satel_to_user * 1000.0)
        log_normal = np.random.normal(0.0, 4.0, [self.num_data_prop, self.total_user])
        Channel_loss = PL_dB + log_normal
        Channel_loss = 10.0 ** -(Channel_loss / 10.0) * 10.0 ** (self.user_antenna_gain / 10.0) / (10.0**(-self.noise/10.0) * self.bandwidth / self.num_user * (10.0**6.0))

        vector_beam = np.zeros([self.num_data_prop, self.num_fixed_beam, 3])
        vector_user = np.zeros([self.num_data_prop, self.total_user, 3])

        vector_beam[:,:,0] = np.squeeze(beam_deployment[:,0] + np.expand_dims(satellite_distance * np.cos(elevation_angle), 1), 1)
        vector_beam[:,:,1] = beam_deployment[:,1]
        vector_beam[:,:,2] = -satellite_distance * np.sin(elevation_angle)

        vector_user[:,:,0] = user_deployment[:,:,0] + satellite_distance * np.cos(elevation_angle)
        vector_user[:,:,1] = user_deployment[:,:,1]
        vector_user[:,:,2] = -satellite_distance * np.sin(elevation_angle)

        theta_denominator = np.expand_dims(np.linalg.norm(vector_beam, axis=-1), -1) * np.expand_dims(np.linalg.norm(vector_user, axis=-1), 1)
        theta_numerator = np.sum(np.expand_dims(vector_beam, 2) * np.expand_dims(vector_user, 1), -1)
        theta_user = np.arccos(theta_numerator / theta_denominator)

        return theta_user, Channel_loss

    def Beam_pattern(self, beta):
        bessel1 = sp.special.j1(beta)
        bessel3 = sp.special.j1(beta) * (8.0 * beta**-2.0 - 1.0) - (4.0 / beta) * sp.special.j0(beta)

        beam_pattern = bessel1 / 2.0 / beta + 36.0 * bessel3 * beta**-3.0
        beam_pattern = beam_pattern**2.0
        beam_pattern = np.where(np.isnan(beam_pattern), 1.0, beam_pattern)
        beam_pattern = np.minimum(beam_pattern, 1.0)

        return beam_pattern

    def Peak_gain(self, dB3_angle):
        peak_gain = 0.5 * (70.0 * np.pi / dB3_angle) ** 2.0

        return peak_gain

    def Beta_generation(self, user_angle, dB3_angle):
        dB3_radian = dB3_angle * np.pi / 180.0
        beta_val = 2.07123 * np.sin(user_angle) / np.sin(dB3_radian)

        return beta_val

    def Channel_gain(self):
        user_angle, channel_loss = self.Generation_channelloss_angle_beampattern()
        beta_data = self.Beta_generation(user_angle, self.dB3_angle)
        beam_pattern = self.Beam_pattern(beta_data) * self.Peak_gain(self.dB3_angle)

        channel_gain = beam_pattern * np.expand_dims(channel_loss, 1)
        channel_gain = np.reshape(channel_gain, [self.num_data_prop, self.num_fixed_beam, self.num_fixed_beam, self.num_user])
        channel_gain = np.swapaxes(channel_gain, 1, 2)

        group_channel_gain = np.tile(np.expand_dims(channel_gain, 1), [1,self.num_group,1,1,1])

        return channel_gain, group_channel_gain

    def Demand_generation(self):
        demand = (np.random.normal(0, 1, [self.num_data_prop, self.total_user]) *
                        self.average_demand / 10.0 * 10.0**6.0 + 10.0**6.0 * self.average_demand) / 10.0**6.0

        demand = np.reshape(demand, [self.num_data_prop, self.num_fixed_beam, self.num_user])
        #demand = np.random.normal(self.average_demand, 1, [self.num_data_prop, self.total_user])
        #print(demand[0])
        #demand = np.ones([self.num_data_prop, self.num_fixed_beam, self.num_user]) * self.average_demand

        return demand

    def snapshot_strategy1(self):
        snapshot = np.tile(np.expand_dims(np.eye(self.num_fixed_beam), 0), [self.num_data_prop, 1, 1])
        group_snapshot = np.sum(np.reshape(snapshot, [self.num_data_prop, self.num_group, self.num_active_beam, self.num_fixed_beam]), axis=2)

        return snapshot, group_snapshot

    def input_DNN(self, group_channel_gain_val, snapshot_index, group_snapshot_index, traffic_demand_data):
        desired_signal = group_channel_gain_val * np.expand_dims(snapshot_index, (1,4)) * np.expand_dims(group_snapshot_index, (2,4))
        inter_signal = group_channel_gain_val * np.expand_dims(group_snapshot_index, (2,4)) * np.expand_dims(group_snapshot_index, (3,4)) - desired_signal

        input_desired_signal = np.reshape(np.sum(desired_signal, (1,3)), [-1,self.num_fixed_beam*self.num_user])
        input_inter_signal = np.reshape(np.swapaxes(np.sum(inter_signal, 1), 2, 3), [-1,self.num_fixed_beam*self.num_user,self.num_fixed_beam])
        input_inter_signal = np.reshape(np.tile(np.expand_dims(input_inter_signal, -1), [1,1,1,self.num_user]), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam * self.num_user])
        eye_element = np.expand_dims(np.tile(np.eye(self.num_user), [self.num_fixed_beam,self.num_fixed_beam]), 0)
        input_inter_signal = input_inter_signal * eye_element

        traffic_demand = np.expand_dims(traffic_demand_data, (1,3)) * np.expand_dims(snapshot_index, (1,4)) * np.expand_dims(group_snapshot_index, (2,4))
        input_traffic_demand = np.reshape(np.sum(traffic_demand, (1,3)), [-1,self.num_fixed_beam*self.num_user]) / self.bandwidth * self.num_timeslot

        return input_desired_signal, input_inter_signal, input_traffic_demand

    def input_DNN2(self, group_channel_gain_val, snapshot_index, group_snapshot_index, traffic_demand_data, time_data_val):
        #group_channel_gain_val = 10.0 * np.log10(group_channel_gain_val)
        desired_signal = group_channel_gain_val * np.expand_dims(snapshot_index, (1, 4)) * np.expand_dims(group_snapshot_index, (2, 4))
        inter_signal = group_channel_gain_val * np.expand_dims(group_snapshot_index, (2, 4)) * np.expand_dims(group_snapshot_index, (3, 4)) - desired_signal

        input_desired_signal = np.reshape(np.sum(desired_signal, (1, 3)), [-1, self.num_fixed_beam * self.num_user])
        input_inter_signal = np.reshape(np.swapaxes(np.sum(inter_signal, 1), 2, 3), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam])
        input_inter_signal = np.reshape(np.tile(np.expand_dims(input_inter_signal, -1), [1,1,1,self.num_user]), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam * self.num_user])
        eye_element = np.expand_dims(np.tile(np.eye(self.num_user), [self.num_fixed_beam,self.num_fixed_beam]), 0)
        input_inter_signal = input_inter_signal * eye_element

        traffic_demand = np.expand_dims(traffic_demand_data / self.bandwidth, (1, 3)) * np.expand_dims(snapshot_index, (1, 4)) * np.expand_dims(group_snapshot_index, (2, 4))
        input_traffic_demand = np.reshape(np.sum(traffic_demand, (1, 3)), [-1, self.num_fixed_beam * self.num_user])

        input_timeslot = (np.tile(np.expand_dims(time_data_val, (2, 3, 4)), [1, 1, self.num_fixed_beam, self.num_fixed_beam, self.num_user])
                          * np.expand_dims(snapshot_index, (1, 4)) * np.expand_dims(group_snapshot_index, (2, 4)))
        input_timeslot = np.reshape(np.sum(input_timeslot, (1,3)), [-1,self.num_fixed_beam*self.num_user])

        input_traffic_demand = input_traffic_demand / input_timeslot

        return input_desired_signal, input_inter_signal, input_traffic_demand, input_timeslot

    def input_DNN3(self, group_channel_gain_val, snapshot_index, group_snapshot_index, traffic_demand_data):
        #group_channel_gain_val = 10.0 * np.log10(group_channel_gain_val)
        desired_signal = group_channel_gain_val * np.expand_dims(snapshot_index, (1, 4)) * np.expand_dims(group_snapshot_index, (2, 4))
        inter_signal = group_channel_gain_val * np.expand_dims(group_snapshot_index, (2, 4)) * np.expand_dims(group_snapshot_index, (3, 4)) - desired_signal

        input_desired_signal = np.reshape(np.sum(desired_signal, (1, 3)), [-1, self.num_fixed_beam * self.num_user])
        input_inter_signal = np.reshape(np.swapaxes(np.sum(inter_signal, 1), 2, 3), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam])
        input_inter_signal = np.reshape(np.tile(np.expand_dims(input_inter_signal, -1), [1,1,1,self.num_user]), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam * self.num_user])
        eye_element = np.expand_dims(np.tile(np.eye(self.num_user), [self.num_fixed_beam,self.num_fixed_beam]), 0)
        input_inter_signal = input_inter_signal * eye_element

        traffic_demand = np.expand_dims(traffic_demand_data, (1,3)) * np.expand_dims(snapshot_index, (1,4)) * np.expand_dims(group_snapshot_index, (2,4))
        input_traffic_demand = np.reshape(np.sum(traffic_demand, (1,3)), [-1,self.num_fixed_beam*self.num_user]) / self.bandwidth

        input_channel = np.sum(group_channel_gain_val * np.expand_dims(group_snapshot_index, (2,4)) * np.expand_dims(group_snapshot_index, (3,4)), 1)
        input_channel = np.reshape(np.sum(np.reshape(input_channel, [-1,self.num_fixed_beam,self.num_group,self.num_active_beam,self.num_user]), 2), [-1,self.num_group,self.num_active_beam**2,self.num_user])
        input_channel = np.reshape(np.swapaxes(input_channel, 2, 3), [-1,self.num_group,self.num_user,self.num_active_beam,self.num_active_beam])

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot
        input_traffic_rate = np.swapaxes(np.reshape(input_traffic_demand, [-1,self.num_group,self.num_active_beam,self.num_user]), 2, 3)
        input_traffic_rate = np.expand_dims(input_traffic_rate, 3) * np.eye(self.num_active_beam)
        input_traffic_value = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.expand_dims(equal_timeslot,(0,2,3,4))) - 1.0

        input_value = np.where(input_traffic_rate == 0, input_channel, input_channel / input_traffic_value)
        input_value = np.log2(1.0 + input_value)

        return input_desired_signal, input_inter_signal, input_traffic_demand, input_channel, input_traffic_rate, input_value

    def Normalized_input(self, group_channel_gain_val, snapshot_index, group_snapshot_index, traffic_demand_data, timeslot_data):
        traffic_demand = np.expand_dims(traffic_demand_data, (1,3)) * np.expand_dims(snapshot_index, (1,4)) * np.expand_dims(group_snapshot_index, (2,4))
        input_traffic_demand = np.reshape(np.sum(traffic_demand, (1,3)), [-1,self.num_fixed_beam*self.num_user]) / self.bandwidth

        input_channel = np.sum(group_channel_gain_val * np.expand_dims(group_snapshot_index, (2,4)) * np.expand_dims(group_snapshot_index, (3,4)), 1)
        input_channel = np.reshape(np.sum(np.reshape(input_channel, [-1,self.num_fixed_beam,self.num_group,self.num_active_beam,self.num_user]), 2), [-1,self.num_group,self.num_active_beam**2,self.num_user])
        input_channel = np.reshape(np.swapaxes(input_channel, 2, 3), [-1,self.num_group,self.num_user,self.num_active_beam,self.num_active_beam])

        input_traffic_rate = np.swapaxes(np.reshape(input_traffic_demand, [-1,self.num_group,self.num_active_beam,self.num_user]), 2, 3)
        input_traffic_rate = np.expand_dims(input_traffic_rate, 3) * np.eye(self.num_active_beam)
        input_traffic_value = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.expand_dims(timeslot_data,(2,3,4))) - 1.0

        input_desired = np.where(input_traffic_rate == 0, 0.0, input_channel / input_traffic_value)
        input_desired = np.sum(input_desired, axis=3)
        input_desired2 = np.where(input_traffic_rate == 0, 0.0, input_channel)
        input_desired2 = np.sum(input_desired2, axis=3)
        input_inter = np.where(input_traffic_rate == 0, input_channel, 0.0)
        input_inter = np.sum(input_inter, axis=3)
        input_state1 = np.reshape(np.log10(1.0 + input_desired/(input_inter + 1.0)), [-1, self.num_group * self.num_user * self.num_active_beam])
        input_state2 = np.reshape(np.log10(1.0 + input_desired), [-1, self.num_group * self.num_user * self.num_active_beam])
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        equal_power = Pmax_val / self.total_user / (self.num_timeslot ** 2.0)
        input_state3 = np.reshape(np.log10(1.0 + (input_inter * equal_power + 1.0) / input_desired), [-1, self.num_group * self.num_user * self.num_active_beam])

        DNN_input1 = np.concatenate((input_state1, timeslot_data / (self.num_timeslot / self.num_group)), axis=-1)
        DNN_input2 = np.concatenate((input_state2, timeslot_data / (self.num_timeslot / self.num_group)), axis=-1)
        DNN_input3 = np.concatenate((input_state3, timeslot_data / (self.num_timeslot / self.num_group)), axis=-1)

        DNN_input3 = input_state3

        return DNN_input1, DNN_input2, DNN_input3

    def capacity_min_max(self, input_traffic_rate, input_channel, timeslot_data, capacity_min, rho_val=0.99):
        capacity_min = np.reshape(capacity_min, [-1, self.total_user])
        current_rate = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.reshape(timeslot_data, [self.num_data_prop, self.num_group, 1, 1, 1])) - 1.0
        timeslot_for_state = np.tile(np.reshape(timeslot_data, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_for_state = np.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
        channel_state = np.where(input_traffic_rate == 0, -input_channel, input_channel / current_rate)
        power_matrix = np.linalg.inv(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_val = np.linalg.eigvals(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_index = np.tile(np.sum(np.where(eig_val < 0.0, 1.0, 0.0), axis=-1, keepdims=True), [1, 1, 1, self.num_active_beam]) > 0.0
        eig_index = np.reshape(np.swapaxes(eig_index, 2, 3), [-1, self.num_fixed_beam, self.num_user])
        power_value = np.swapaxes(np.sum(power_matrix, -1), 2, 3)
        equal_power = np.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])

        additional_val = np.minimum(np.sum(channel_state, 4), 0.0) * -1.0
        additional_val = np.expand_dims(additional_val, 4)
        addi_index = np.tile(additional_val, [1, 1, 1, 1, self.num_active_beam]) > 0.0
        desired = np.where(input_traffic_rate == 0, 0.0, input_channel)
        inter_sum = np.sum(input_channel - desired, axis=4, keepdims=True)
        addi_val = np.where(addi_index, (desired / inter_sum) * rho_val, current_rate)
        addi_val = np.where(addi_val > current_rate, current_rate, addi_val)
        addi_val = np.swapaxes(np.sum(addi_val, -1), 2, 3)
        addi_val = np.reshape(addi_val, [-1, self.total_user])
        desired = np.where(input_traffic_rate == 0, 0.0, input_channel)
        inter_sum = np.sum(input_channel - desired, axis=4, keepdims=True)
        capacity_max = np.log2((desired / inter_sum) * rho_val + 1.0)
        capacity_max = np.reshape(np.swapaxes(np.sum(capacity_max, -1), 2, 3), [-1, self.total_user]) / self.num_user / self.num_timeslot * timeslot_for_state
        capacity_min = capacity_min / self.bandwidth

        return capacity_min, capacity_max

    def data_rate_proposed(self, group_channel_gain_val, group_power_val, snapshot_index, group_snapshot_index, time_data_val):
        channel_power_val = group_channel_gain_val * group_power_val
        SINR_desired = np.sum(channel_power_val * np.expand_dims(snapshot_index, (1,4)) * np.expand_dims(group_snapshot_index, (2,4)), 3)
        SINR_inter = np.sum(channel_power_val * np.expand_dims(group_snapshot_index, (2,4)) * np.expand_dims(group_snapshot_index, (3,4)), 3) - SINR_desired

        SINR_val = SINR_desired / (SINR_inter + 1.0)

        data_rate = np.where(np.tile(np.expand_dims(group_snapshot_index, -1), (1,1,1,self.num_user)), np.expand_dims(time_data_val, (2,3)) * np.log2(1.0 + SINR_val), 0)
        data_rate = self.bandwidth / self.num_user / self.num_timeslot * np.sum(data_rate, 1)

        return data_rate

    def energy_efficiency(self, capacity, power_alloc, time_alloc, group_index):
        total_power = np.expand_dims(power_alloc, 1) * np.expand_dims(time_alloc, (2, 3)) * np.expand_dims(group_index, -1)
        total_power = (np.sum(total_power, (1, 2, 3)) / 1000.0 + 1.0) / self.num_timeslot
        sum_capacity = np.sum(capacity, (1, 2))
        EE = sum_capacity / total_power

        return EE

    def outage_rate(self, capacity, traffic_demand):
        capacity = np.reshape(capacity, [-1, self.total_user])
        traffic_demand = np.reshape(traffic_demand, [-1, self.total_user])
        outage = np.sum(np.where(capacity >= (traffic_demand * 0.99), 1, 0), axis=-1)
        outage = np.where(outage >= self.total_user, 0, 1)
        outage = np.sum(outage) / self.num_data_prop * 100.0

        return outage

    def closed_form_PAC_instance(self, group_channel_gain_val, group_power_val, snapshot_index, group_snapshot_index, time_data_val, traffic_demand_data):
        channel_power_val = group_channel_gain_val * group_power_val

        traffic_demand_rate = 2.0 ** (np.reshape(traffic_demand_data, [1, self.num_fixed_beam, self.num_user]) * self.num_timeslot / (self.bandwidth / self.num_user) / np.expand_dims(time_data_val, (1,2))) - 1.0

        desired_signal = np.sum(channel_power_val * np.expand_dims(snapshot_index, (0,3)) * np.expand_dims(group_snapshot_index, (1,3)), 2)
        inter_signal = np.sum(channel_power_val * np.expand_dims(group_snapshot_index, (1,3)) * np.expand_dims(group_snapshot_index, (2,3)), 2) - desired_signal + 1.0
        desired_signal2 = np.sum(group_channel_gain_val * np.expand_dims(snapshot_index, (0,3)) * np.expand_dims(group_snapshot_index, (1,3)), 2)

        power_allocation_coefficient = np.where(np.tile(np.expand_dims(group_snapshot_index, -1), (1,1,self.num_user)), traffic_demand_rate * inter_signal / desired_signal2, 0)
        power_allocation_coefficient = np.sum(power_allocation_coefficient, 0)

        return power_allocation_coefficient

    def matrix_inversion_based_power_allocation(self, input_traffic_rate, input_channel, timeslot_data, rho_val=0.99):
        current_rate = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.reshape(timeslot_data, [self.num_data_prop, self.num_group, 1, 1, 1])) - 1.0
        channel_state = np.where(input_traffic_rate == 0, -input_channel, input_channel / current_rate)
        power_matrix = np.linalg.inv(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_val = np.linalg.eigvals(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_index = np.tile(np.sum(np.where(eig_val < 0.0, 1.0, 0.0), axis=-1, keepdims=True), [1, 1, 1, self.num_active_beam]) > 0.0
        eig_index = np.reshape(np.swapaxes(eig_index, 2, 3), [-1, self.num_fixed_beam, self.num_user])
        power_value = np.swapaxes(np.sum(power_matrix, -1), 2, 3)
        equal_power = np.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])

        additional_val = np.minimum(np.sum(channel_state, 4), 0.0) * -1.0
        additional_val = np.expand_dims(additional_val, 4)
        addi_index = np.tile(additional_val, [1, 1, 1, 1, self.num_active_beam]) > 0.0
        desired = np.where(input_traffic_rate == 0, 0.0, input_channel)
        inter_sum = np.sum(input_channel - desired, axis=4, keepdims=True)
        addi_val = np.where(addi_index, (desired / inter_sum) * rho_val, current_rate)
        addi_val = np.where(addi_val > current_rate, current_rate, addi_val)
        channel_state = np.where(input_traffic_rate == 0, -input_channel, input_channel / addi_val)
        power_matrix = np.linalg.inv(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        power_value = np.swapaxes(np.sum(power_matrix, -1), 2, 3)
        eqp = np.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])
        equal_power = np.where(eig_index, eqp, equal_power)
        power_time = np.reshape(np.tile(np.expand_dims(timeslot_data, (2, 3)), [1, 1, self.num_active_beam, self.num_user]), [-1, self.num_fixed_beam, self.num_user])
        equal_power = equal_power * power_time / self.num_timeslot
        equal_power = np.where(np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)) <= 10.0 ** (self.maximum_transmit_power / 10.0),
                               equal_power, equal_power * 10.0 ** (self.maximum_transmit_power / 10.0) / np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)))
        equal_power = equal_power / power_time * self.num_timeslot
        equal_power = np.where(np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)) <= 10.0 ** (self.maximum_transmit_power / 10.0),
                               equal_power, equal_power * 10.0 ** (self.maximum_transmit_power / 10.0) / np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)))

        matrix_inverse_based_power_allocation = equal_power

        return matrix_inverse_based_power_allocation

    def matrix_inversion_based_power_allocation2(self, input_traffic_rate, input_channel, timeslot_data, rho_val=0.99):
        current_rate = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.reshape(timeslot_data, [self.num_data_prop, self.num_group, 1, 1, 1])) - 1.0
        channel_state = np.where(input_traffic_rate == 0, -input_channel, input_channel / current_rate)
        power_matrix = np.linalg.inv(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_val = np.linalg.eigvals(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_index = np.tile(np.sum(np.where(eig_val < 0.0, 1.0, 0.0), axis=-1, keepdims=True), [1, 1, 1, self.num_active_beam]) > 0.0
        eig_index = np.reshape(np.swapaxes(eig_index, 2, 3), [-1, self.num_fixed_beam, self.num_user])
        power_value = np.swapaxes(np.sum(power_matrix, -1), 2, 3)
        equal_power = np.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])

        channel_state = np.where(input_traffic_rate == 0, -input_channel, input_channel / current_rate)
        power_matrix = np.linalg.inv(np.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        power_value = np.swapaxes(np.sum(power_matrix, -1), 2, 3)
        eqp = np.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])
        equal_power = np.where(eig_index, eqp, equal_power)
        power_time = np.reshape(np.tile(np.expand_dims(timeslot_data, (2, 3)), [1, 1, self.num_active_beam, self.num_user]), [-1, self.num_fixed_beam, self.num_user])
        equal_power = equal_power * power_time / self.num_timeslot
        equal_power = np.where(np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)) <= 10.0 ** (self.maximum_transmit_power / 10.0),
                               equal_power, equal_power * 10.0 ** (self.maximum_transmit_power / 10.0) / np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)))
        equal_power = equal_power / power_time * self.num_timeslot
        equal_power = np.where(np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)) <= 10.0 ** (self.maximum_transmit_power / 10.0),
                               equal_power, equal_power * 10.0 ** (self.maximum_transmit_power / 10.0) / np.expand_dims(np.sum(equal_power, (1, 2)), (1, 2)))

        matrix_inverse_based_power_allocation = equal_power

        return matrix_inverse_based_power_allocation

    def time_allocation_algorithm(self, traffic_demand_data, snapshot_index, group_snapshot_index, group_channel_gain_val, maximum_power_val):
        time_data_cal = np.ones(self.num_group)

        while True:
            if (self.num_timeslot - np.sum(time_data_cal)) == 0:
                break
            objective_cal = []
            time_data_list = []
            time_data_cal = time_data_cal + 0
            for time_data in range(self.num_group):
                time_data_cal2 = time_data_cal + 0
                time_data_cal2[time_data] = time_data_cal2[time_data] + 1

                Opt_power = self.closed_form_PAC_instance(group_channel_gain_val, maximum_power_val / self.total_user, snapshot_index, group_snapshot_index, time_data_cal2, traffic_demand_data)

                objective_val = np.sum(np.expand_dims(Opt_power, 0) * np.expand_dims(time_data_cal2, (1,2)) * np.expand_dims(group_snapshot_index, -1)) / self.num_timeslot

                objective_cal.append(objective_val)
                time_data_list.append(time_data_cal2)

            objective_argmin = np.argmin(objective_cal)
            objective_min_data = objective_cal[objective_argmin]
            time_data_cal = time_data_list[objective_argmin]

            if (self.num_timeslot - np.sum(time_data_cal)) == 0:
                break

        return time_data_cal

    def PA_without_US(self, time_data_cal, traffic_demand_data, snapshot_index, group_snapshot_index, group_channel_gain_val):
        end = True
        iter_time = 0
        epsilon = 0.0001
        traffic_demand_rate = 2.0 ** (np.reshape(traffic_demand_data, [1, self.num_fixed_beam, self.num_user]) * self.num_timeslot / (self.bandwidth / self.num_user) / np.expand_dims(time_data_cal, (1, 2))) - 1.0

        while end == True:
            if iter_time == 0:
                power_val = np.ones([self.num_fixed_beam,self.num_user]) * 0.01
                pre_alloc = np.zeros([self.num_fixed_beam,self.num_user]) * 0.01
                check_alloc = np.zeros([self.num_fixed_beam,self.num_user])
            else:
                power_val = power_val + 0.0
                pre_alloc = pre_alloc + 0.0
                check_alloc = power_val + 0.0

            for beam in range(self.num_fixed_beam):
                group_power_val = np.tile(np.expand_dims(power_val, (0,1)), [self.num_group, self.num_fixed_beam, 1, 1])
                Opt_power = self.closed_form_PAC_instance(group_channel_gain_val, group_power_val, snapshot_index, group_snapshot_index, time_data_cal, traffic_demand_data)
                time_data_cal2 = np.expand_dims(np.sum(np.mean(np.expand_dims(time_data_cal, (1,2)) * np.ones([self.num_group,self.num_fixed_beam,self.num_user]), -1) * group_snapshot_index, 0), -1)
                Opt_power = np.minimum(Opt_power, 10.0 ** (self.maximum_transmit_power/10.0) / self.total_user)

                pre_alloc[beam, :] = Opt_power[beam, :] + 0.0
                power_val[beam, :] = np.where(np.isfinite(power_val[beam, :]), pre_alloc[beam, :] + 0.0, 0.0)

            old_new = np.sqrt(np.sum(np.abs(power_val ** 2.0 - check_alloc ** 2.0)))
            #old_new = np.sum(np.abs(power_val - check_alloc))
            iter_time = iter_time + 1

            if iter_time > 100:
                end = False
                break

            if old_new <= epsilon:
                end = False
                break

        return power_val, traffic_demand_rate

    def Proposed_Heuristic_Algorithm(self, channel_gain, traffic_demand, snapshot_index, group_snapshot_index):
        power_allocation_opt = np.zeros([self.num_data_prop, self.num_fixed_beam, self.num_user])
        time_allocation_opt = np.zeros([self.num_data_prop, self.num_group])

        maximum_power_val = 10.0 ** (self.maximum_transmit_power / 10.0)

        for data in range(self.num_data_prop):
            start_time = time.time()
            channel_gain_data = channel_gain[data]
            traffic_demand_data = traffic_demand[data]
            group_snapshot_data = group_snapshot_index[data]
            snapshot_data = snapshot_index[data]

            time_data_cal = self.time_allocation_algorithm(traffic_demand_data, snapshot_data, group_snapshot_data, channel_gain_data, maximum_power_val)
            power_val, traffic_demand_rate = self.PA_without_US(time_data_cal, traffic_demand_data, snapshot_data, group_snapshot_data, channel_gain_data)

            group_power_val = np.tile(np.expand_dims(power_val, (0,1)), [self.num_group,self.num_fixed_beam,1,1])
            channel_power_val = channel_gain_data * group_power_val

            SINR_desired = np.sum(channel_power_val * np.expand_dims(snapshot_data, (0,3)) * np.expand_dims(group_snapshot_data, (1,3)), 2)
            SINR_inter = np.sum(channel_power_val * np.expand_dims(group_snapshot_data, (1,3)) * np.expand_dims(group_snapshot_data, (2,3)), 2) - SINR_desired

            SINR_val = SINR_desired / (SINR_inter + 1.0)

            data_rate = np.where(np.tile(np.expand_dims(group_snapshot_data, -1), (1,1,self.num_user)), np.expand_dims(time_data_cal, (1,2)) * np.log2(1.0 + SINR_val), 0)
            data_rate = self.bandwidth / self.num_user / self.num_timeslot * np.sum(data_rate, 0)

            power_allocation_opt[data] = power_val
            time_allocation_opt[data] = time_data_cal

            total_power = np.expand_dims(power_val, 0) * np.expand_dims(time_data_cal, (1,2)) * np.expand_dims(group_snapshot_data, -1) / self.num_timeslot
            end_time = time.time() - start_time

            print('###############################################################')
            print('Proposed Heuristic Algorithm: num_user ' + str(self.total_user) + ' average_demand ' + str(self.average_demand) + '[Mbps]')
            print('data number: ' + str(data) + ', computational time : ' + str(end_time))
            print('Power consumption: ' + str(np.sum(total_power)))
            print('Capacity-demand gap: '+ str(np.sum(np.abs(data_rate - traffic_demand_data))))
            print('Per-group timeslot allocation: ')
            print(time_data_cal)
            print('Capacity: ')
            print(np.reshape(data_rate, self.num_fixed_beam*self.num_user))
            print('Requsted traffic demand: ')
            print(np.reshape(traffic_demand_data, self.num_fixed_beam*self.num_user))
            print('###############################################################')
            print('\n')

        return power_allocation_opt, time_allocation_opt

    def Conventional_Heuristic_Algorithm(self, channel_gain, traffic_demand, snapshot_index, group_snapshot_index):
        power_allocation_opt = np.zeros([self.num_data_prop, self.num_timeslot, self.num_fixed_beam, self.num_user])
        time_allocation_opt = np.zeros([self.num_data_prop, self.num_timeslot, self.num_group])

        maximum_power_val = 10.0 ** (self.maximum_transmit_power / 10.0)

        I2 = 30

        for data in range(self.num_data_prop):
            start_time = time.time()
            channel_gain_data = channel_gain[data]
            traffic_demand_data = traffic_demand[data]
            group_snapshot_data = group_snapshot_index[data]
            snapshot_data = snapshot_index[data]

            power_data_iter = np.zeros([self.num_timeslot, self.num_fixed_beam, self.num_user])
            schedule_data_iter = np.zeros([self.num_timeslot,self.num_group])
            data_rate_iter = np.zeros([self.num_timeslot,self.num_fixed_beam,self.num_user])

            channel_gain_data_iter = np.tile(np.expand_dims(channel_gain_data, 0), [self.num_timeslot,1,1,1])
            rate_already_allocated = np.zeros([self.num_fixed_beam,self.num_user])
            user_not_allocated = np.ones([self.num_fixed_beam,self.num_user])

            init_power = np.ones([self.num_fixed_beam, self.num_user]) * maximum_power_val / self.total_user

            for time_data in range(self.num_timeslot):
                power_previous = np.tile(np.expand_dims(init_power, (0,1)), [self.num_group,self.num_fixed_beam,1,1])
                channel_data = channel_gain_data_iter[time_data] + 0.0
                channel_power_val = channel_data * power_previous

                gamma_val_desired = np.sum(channel_power_val * np.expand_dims(snapshot_data, (0, 3)) * np.expand_dims(group_snapshot_data, (1, 3)), 2)
                gamma_val_inter = np.sum(channel_power_val * np.expand_dims(group_snapshot_data, (1, 3)) * np.expand_dims(group_snapshot_data, (2, 3)), 2) - gamma_val_desired + 1.0
                gamma_val = gamma_val_desired / gamma_val_inter

                eta_val_desired = self.bandwidth / self.num_user / self.num_timeslot * np.sqrt((1.0 + gamma_val) * gamma_val_desired)
                eta_val_inter = gamma_val_inter + 1.0
                eta_val = eta_val_desired / eta_val_inter

                rate_cal1 = self.bandwidth / self.num_user / self.num_timeslot * (np.log2(1.0 + gamma_val) - gamma_val)
                rate_cal2 = 2.0 * eta_val * np.sqrt(self.bandwidth / self.num_user / self.num_timeslot * (1.0 + gamma_val) * gamma_val_desired)
                rate_cal3 = eta_val ** 2.0 * eta_val_inter

                rate_data = rate_cal1 + rate_cal2 - rate_cal3
                scheduling_objective = np.expand_dims(traffic_demand_data, 0) - rate_data - np.expand_dims(rate_already_allocated, 0)
                scheduling_objective = scheduling_objective + np.maximum(self.R_min - rate_data - np.expand_dims(rate_already_allocated, 0), 0.0)
                scheduling_objective = np.expand_dims(user_not_allocated, 0) * np.expand_dims(group_snapshot_data, -1) * scheduling_objective * np.expand_dims(group_snapshot_data, -1)
                scheduling_objective = np.where(np.isfinite(scheduling_objective), scheduling_objective, 0.0)
                scheduling_objective = np.maximum(scheduling_objective, 0.0)
                scheduling_objective = np.sum(scheduling_objective, (1,2))
                scheduling_objective = np.where(scheduling_objective == 0.0, np.infty, scheduling_objective)

                schedule_previous = np.min(scheduling_objective, keepdims=True) == scheduling_objective

                for iter2 in range(I2):
                    desired_channel = np.sum(channel_data * np.expand_dims(snapshot_data, (0, 3)) * np.expand_dims(group_snapshot_data, (1, 3)), 2)
                    power_val_numerator = 1.0 ** 2.0 * eta_val ** 2.0 * self.bandwidth / self.num_user / self.num_timeslot * (1.0 + gamma_val) * desired_channel
                    power_val_denominator = np.sum(desired_channel * 2.0 * eta_val ** 2.0, axis=1, keepdims=True) ** 2.0
                    power_val_immediate = power_val_numerator / power_val_denominator
                    power_val_immediate = maximum_power_val / self.num_fixed_beam * np.expand_dims(group_snapshot_data, -1) / self.num_timeslot

                    power_val = power_val_immediate

                    channel_power_val = np.expand_dims(channel_data, 0) * np.tile(np.expand_dims(power_val, 1), [1,self.num_fixed_beam,1,1])

                    gamma_val_desired = np.sum(channel_power_val * np.expand_dims(snapshot_data, (0, 3)) * np.expand_dims(group_snapshot_data, (1, 3)), 2)
                    gamma_val_inter = np.sum(channel_power_val * np.expand_dims(group_snapshot_data, (1, 3)) * np.expand_dims(group_snapshot_data, (2, 3)), 2) - gamma_val_desired + 1.0
                    gamma_val = gamma_val_desired / gamma_val_inter

                    eta_val_desired = self.bandwidth / self.num_user / self.num_timeslot * np.sqrt((1.0 + gamma_val) * gamma_val_desired)
                    eta_val_inter = gamma_val_inter + 1.0
                    eta_val = eta_val_desired / eta_val_inter

                    rate_cal1 = self.bandwidth / self.num_user / self.num_timeslot * (np.log2(1.0 + gamma_val) - gamma_val)
                    rate_cal2 = 2.0 * eta_val * np.sqrt(self.bandwidth / self.num_user / self.num_timeslot * (1.0 + gamma_val) * gamma_val_desired)
                    rate_cal3 = eta_val ** 2.0 * eta_val_inter

                    rate_data = rate_cal1 + rate_cal2 - rate_cal3
                    scheduling_objective = np.expand_dims(traffic_demand_data, 0) - rate_data - np.expand_dims(rate_already_allocated, 0)
                    scheduling_objective = scheduling_objective + 100.0 * np.maximum(self.R_min - rate_data - np.expand_dims(rate_already_allocated, 0), 0.0)
                    scheduling_objective = np.expand_dims(user_not_allocated, 0) * np.expand_dims(group_snapshot_data, -1) * scheduling_objective * np.expand_dims(group_snapshot_data, -1)
                    scheduling_objective = np.where(np.isfinite(scheduling_objective), scheduling_objective, 0.0)
                    #scheduling_objective = np.maximum(scheduling_objective, 0.0)
                    scheduling_objective = np.sum(scheduling_objective, (1, 2))
                    scheduling_objective = np.where(scheduling_objective == 0.0, np.infty, scheduling_objective)

                    scheduling_index = np.min(scheduling_objective, keepdims=True) == scheduling_objective

                    power_variation = np.sum(np.abs(power_previous - np.expand_dims(power_val, 1)))
                    scheduling_variation = np.sum(np.abs(schedule_previous - 1.0 * scheduling_index))
                    total_variation = power_variation + scheduling_variation

                    schedule_previous = scheduling_index
                    power_previous = np.sum(power_val, 0)

                    if total_variation == 0:
                        break

                #print(scheduling_index)

                power_val = power_val * np.expand_dims(scheduling_index, (1,2))
                power_init = power_val + 0.0
                i = 0
                while True:
                    channel_power_val = np.expand_dims(channel_data, 0) * np.tile(np.expand_dims(power_init, 1), [1, self.num_fixed_beam, 1, 1])
                    gamma_val_desired = np.sum(channel_power_val * np.expand_dims(snapshot_data, (0, 3)) * np.expand_dims(group_snapshot_data, (1, 3)), 2)
                    gamma_val_inter = np.sum(channel_power_val * np.expand_dims(group_snapshot_data, (1, 3)) * np.expand_dims(group_snapshot_data, (2, 3)), 2) - gamma_val_desired + 1.0
                    gamma_val = gamma_val_desired / gamma_val_inter
                    data_rate = self.bandwidth / self.num_user * np.log2(1.0 + gamma_val) * np.expand_dims(scheduling_index, (1,2))

                    rate_immediate = rate_already_allocated + np.sum(data_rate, 0)
                    rate_over_index = (rate_immediate > traffic_demand_data) * np.expand_dims(scheduling_index, (1,2))
                    rate_fitted = (traffic_demand_data - rate_already_allocated) * np.expand_dims(scheduling_index, (1,2))

                    #print(rate_fitted)

                    power_fitted = (2.0 ** (rate_fitted / self.bandwidth * self.num_user * self.num_timeslot) - 1.0) * gamma_val_inter / desired_channel
                    power_fitted = np.where(desired_channel > 0, power_fitted, 0.0)
                    power_fitted = np.where(power_fitted > maximum_power_val, power_val, power_fitted)
                    power_fitted = np.where(power_fitted < 0, 0.0, power_fitted)
                    power_val = np.where(rate_over_index == 1, power_fitted, power_init)
                    i = i + 1

                    if np.sum(power_init - power_val) == 0:
                        power_val = power_init
                        break
                    elif i >= 100:
                        break
                    else:
                        power_init = power_val + 0.0

                channel_power_val = np.expand_dims(channel_data, 0) * np.tile(np.expand_dims(power_val, 1), [1, self.num_fixed_beam, 1, 1])
                gamma_val_desired = np.sum(channel_power_val * np.expand_dims(snapshot_data, (0, 3)) * np.expand_dims(group_snapshot_data, (1, 3)), 2)
                gamma_val_inter = np.sum(channel_power_val * np.expand_dims(group_snapshot_data, (1, 3)) * np.expand_dims(group_snapshot_data, (2, 3)), 2) - gamma_val_desired + 1.0
                gamma_val = gamma_val_desired / gamma_val_inter
                data_rate = self.bandwidth / self.num_user / self.num_timeslot * np.log2(1.0 + gamma_val) * np.expand_dims(scheduling_index, (1, 2))

                rate_already_allocated = rate_already_allocated + np.sum(data_rate, 0)
                user_not_allocated = np.maximum(user_not_allocated - (rate_already_allocated >= traffic_demand_data - 0.00001)*1.0, 0)

                power_data_iter[time_data] = np.sum(power_val, 0)
                schedule_data_iter[time_data] = scheduling_index
                data_rate_iter[time_data] = np.sum(data_rate, 0)

                if np.sum(user_not_allocated) == 0:
                    break

            time_data_cal = np.sum(schedule_data_iter, 0)
            power_val = np.sum(power_data_iter, 0)

            power_allocation_opt[data] = power_data_iter
            time_allocation_opt[data] = schedule_data_iter

            total_power = np.sum(power_data_iter)
            end_time = time.time() - start_time

            print('###############################################################')
            print('Conventional Heuristic Algorithm: num_user ' + str(self.total_user) + ' average_demand ' + str(self.average_demand) + '[Mbps]')
            print('data number: ' + str(data) + ', computational time : ' + str(end_time))
            print('Power consumption: ' + str(np.sum(total_power)))
            print('Capacity-demand gap: '+ str(np.sum(np.abs(np.sum(data_rate_iter, 0) - traffic_demand_data))))
            print('Per-group timeslot allocation: ')
            print(time_data_cal)
            print('Capacity: ')
            print(np.reshape(np.sum(data_rate_iter, 0), self.num_fixed_beam*self.num_user))
            print('Requsted traffic demand: ')
            print(np.reshape(traffic_demand_data, self.num_fixed_beam*self.num_user))
            print('###############################################################')
            print('\n')

        return power_allocation_opt, time_allocation_opt

    def capacity_for_IPM(self, desired, inter, timeslot, power, Pmax_val):
        power = power * Pmax_val
        desired_SNR = desired * power
        inter_SNR = np.sum(inter * np.expand_dims(power, 0), -1) + 1.0
        capacity = self.bandwidth / self.num_user / self.num_timeslot * timeslot * np.log2(1.0 + desired_SNR / inter_SNR)

        return capacity

    def capacity_derv(self, desired, inter, timeslot, power, Pmax_val):
        power = power * Pmax_val
        desired_SNR = desired * power
        inter_SNR = np.sum(inter * np.expand_dims(power, 0), -1) + 1.0

        desired_derv1 = self.bandwidth / self.num_user * timeslot / self.num_timeslot
        desired_derv2 = desired / inter_SNR
        desired_derv3 = (1.0 + desired_SNR / inter_SNR) * np.log(2.0)
        desired_derv = desired_derv1 * desired_derv2 / desired_derv3

        inter_index = np.where(inter == 0.0, 0.0, 1.0)
        inter_desired = inter_index * np.expand_dims(desired, 0)
        inter_inter = inter_index * np.expand_dims(inter_SNR, 0)

        inter_derv1 = self.bandwidth / self.num_user * timeslot / self.num_timeslot
        inter_derv2 = 1.0 + inter_desired * power / inter_inter
        inter_derv3 = np.transpose(inter) * inter_desired * power / inter_inter ** 2.0
        inter_derv = np.sum(np.where(inter_index == 1, inter_derv1 * inter_derv3 / inter_derv2, 0.0), -1)

        derv_capacity = desired_derv - inter_derv

        return derv_capacity

    def EE_derv(self, desired, inter, timeslot, power, Pmax_val):
        power = power * Pmax_val
        total_power = (np.sum(power * timeslot) / 1000.0 + 1.0)
        desired_SNR = desired * power
        inter_SNR = np.sum(inter * np.expand_dims(power, 0), -1) + 1.0
        capacity = self.bandwidth / self.num_user * timeslot * np.log2(1.0 + desired_SNR / inter_SNR)

        desired_derv1 = self.bandwidth / self.num_user * timeslot
        desired_derv2 = desired / inter_SNR
        desired_derv3 = (1.0 + desired_SNR / inter_SNR) * np.log(2.0)
        desired_derv = desired_derv1 * desired_derv2 / desired_derv3

        inter_index = np.where(inter == 0.0, 0.0, 1.0)
        inter_desired = inter_index * np.expand_dims(desired, 0)
        inter_inter = inter_index * np.expand_dims(inter_SNR, 0)

        inter_derv1 = self.bandwidth / self.num_user * timeslot
        inter_derv2 = 1.0 + inter_desired * power / inter_inter
        inter_derv3 = np.transpose(inter) * inter_desired * power / inter_inter ** 2.0
        inter_derv = np.sum(np.where(inter_index == 1, inter_derv1 * inter_derv3 / inter_derv2, 0.0), 0)

        derv_capacity = (desired_derv - inter_derv) * Pmax_val
        derv_EE = (derv_capacity * total_power - np.sum(capacity) * timeslot * Pmax_val / 1000.0) / (total_power ** 2.0)

        return derv_EE

    def constraint_linear(self, desired, inter, timeslot, Pmax_val, traffic_rate):
        traffic_value = 2.0 ** (traffic_rate * self.num_user * self.num_timeslot / timeslot) - 1.0
        desired_part = np.eye(self.total_user) * desired * Pmax_val / traffic_value
        inter_part = -inter * Pmax_val
        constraint = desired_part + inter_part

        return constraint

    def EE_for_IPM(self, desired, inter, timeslot, power, Pmax_val):
        power = power * Pmax_val
        desired_SNR = desired * power
        inter_SNR = np.sum(inter * np.expand_dims(power, 0), -1) + 1.0
        capacity = self.bandwidth / self.num_user * timeslot * np.log2(1.0 + desired_SNR / inter_SNR)
        total_power = np.sum(power * timeslot)
        EE = np.sum(capacity) / (total_power / 1000.0 + 1.0)

        return EE

    def TR_IPM_PA_algorithm_perslot(self, desired_signal, inter_signal, traffic_demand, channel_gain, real_traffic, timeslot_alloc, minimum_power, keep_train):
        from ipsolver import minimize_constrained, NonlinearConstraint, LinearConstraint, BoxConstraint

        savefile_str = ('Altitude' + str(self.Altitude) + 'active_beam' + str(self.num_active_beam) +
                        'fixed_beam' + str(self.num_fixed_beam) + 'user' + str(self.num_user) + 'timeslot' + str(self.num_timeslot) +
                        'elevation_angle' + str(self.num_angle) + 'Tx_power' + str(self.maximum_transmit_power) +
                        'average_demand' + str(self.average_demand) + 'bandwidth' + str(self.bandwidth)
                        )

        IPM_opt = np.zeros([self.num_data_prop, self.total_user])
        IPM_time = np.zeros(self.num_data_prop)
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)

        timeslot_alloc = np.tile(np.reshape(timeslot_alloc, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])

        power_const = np.eye(self.total_user)

        if keep_train == 1:
            for data in range(self.num_data_prop):
            #for data in range(983,self.num_data_prop):
                start_time = time.time()

                desired_data = desired_signal[data]
                inter_data = inter_signal[data]
                demand_data = traffic_demand[data] * self.bandwidth
                channel_data = channel_gain[data]
                real_data = real_traffic[data]
                timeslot_data = timeslot_alloc[data]
                minimum_data = minimum_power[data]

                # x0 = np.zeros([self.total_user])
                x0 = minimum_data / Pmax_val

                fun = lambda x: -np.sum(self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val))
                grad = lambda x: -self.EE_derv(desired_data, inter_data, timeslot_data, x, Pmax_val)

                constraint = self.constraint_linear(desired_data, inter_data, timeslot_data, Pmax_val, demand_data / self.bandwidth)

                linear1 = LinearConstraint(np.ones(self.total_user) * timeslot_data / self.num_timeslot, ("less", 1.0))
                linear2 = LinearConstraint(constraint, ("greater", np.ones(self.total_user) * 1.0))
                # box = BoxConstraint(("interval", 0.0, 1.0))
                linear3 = LinearConstraint(power_const, ("interval", x0, np.ones(self.total_user)))

                result = minimize_constrained(fun, x0, grad, hess='cs', constraints=(linear1, linear2, linear3), method='tr_interior_point',
                                              sparse_jacobian=True, xtol=1e-8, gtol=1e-8, max_iter=100, verbose=0)
                x = result.x
                x = np.where(x > 0, x, 0)

                capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                total_power = np.sum(x * Pmax_val * timeslot_data / self.num_timeslot)

                minimum_capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_outage = np.sum(minimum_capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                minimum_total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data)

                if ((outage == True) and (minimum_outage == False)) == True:
                    capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                    total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data / self.num_timeslot)
                    x = minimum_data / Pmax_val

                end_time = time.time() - start_time

                print('###############################################################')
                print('Trust region IPM (per-slot): num_user ' + str(self.total_user) + ' average_demand ' + str(self.average_demand) + '[Mbps]')
                print('data number: ' + str(data) + ', computational time : ' + str(end_time))
                print('Total capacity: ', np.sum(capacity))
                print(capacity)
                print('Energy efficiency: ', EE)
                print('minimum outage occurs: ', minimum_outage)
                print('outage occurs: ', outage)
                print('total power consumption: ', 10.0 * np.log10(total_power))
                print('###############################################################')
                print('\n')

                IPM_opt[data] = x * Pmax_val
                IPM_time[data] = end_time

            # Save for per-slot-based timeslot
            np.savetxt('./IPM_data/perslot/power_val/' + str(savefile_str) + '.csv', IPM_opt, delimiter=",")
            np.savetxt('./IPM_data/perslot/execute_time/' + str(savefile_str) + '.csv', IPM_time, delimiter=",")

        if keep_train == 0:
            # Load for per-slot-based timeslot
            IPM_opt = np.loadtxt('./IPM_data/perslot/power_val/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)
            IPM_time = np.loadtxt('./IPM_data/perslot/execute_time/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)

        return IPM_opt, IPM_time

    def TR_IPM_PA_algorithm_iterative(self, desired_signal, inter_signal, traffic_demand, channel_gain, real_traffic, timeslot_alloc, minimum_power, keep_train):
        from ipsolver import minimize_constrained, NonlinearConstraint, LinearConstraint, BoxConstraint

        savefile_str = ('Altitude' + str(self.Altitude) + 'active_beam' + str(self.num_active_beam) +
                        'fixed_beam' + str(self.num_fixed_beam) + 'user' + str(self.num_user) + 'timeslot' + str(self.num_timeslot) +
                        'elevation_angle' + str(self.num_angle) + 'Tx_power' + str(self.maximum_transmit_power) +
                        'average_demand' + str(self.average_demand) + 'bandwidth' + str(self.bandwidth)
                        )

        IPM_opt = np.zeros([self.num_data_prop, self.total_user])
        IPM_time = np.zeros(self.num_data_prop)
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)

        timeslot_alloc = np.tile(np.reshape(timeslot_alloc, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])

        power_const = np.eye(self.total_user)

        if keep_train == 1:
            for data in range(self.num_data_prop):
                start_time = time.time()

                desired_data = desired_signal[data]
                inter_data = inter_signal[data]
                demand_data = traffic_demand[data] * self.bandwidth
                channel_data = channel_gain[data]
                real_data = real_traffic[data]
                timeslot_data = timeslot_alloc[data]
                minimum_data = minimum_power[data]

                #x0 = np.zeros([self.total_user])
                x0 = minimum_data / Pmax_val

                fun = lambda x: -np.sum(self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val))
                grad = lambda x: -self.EE_derv(desired_data, inter_data, timeslot_data, x, Pmax_val)

                constraint = self.constraint_linear(desired_data, inter_data, timeslot_data, Pmax_val, demand_data / self.bandwidth)

                linear1 = LinearConstraint(np.ones(self.total_user) * timeslot_data / self.num_timeslot, ("less", 1.0))
                linear2 = LinearConstraint(constraint, ("greater", np.ones(self.total_user)*1.0))
                # box = BoxConstraint(("interval", 0.0, 1.0))
                linear3 = LinearConstraint(power_const, ("interval", x0, np.ones(self.total_user)))

                result = minimize_constrained(fun, x0, grad, hess='cs', constraints=(linear1, linear2, linear3), method='tr_interior_point',
                                              sparse_jacobian=True, xtol=1e-8, gtol=1e-8, max_iter=100, verbose=0)
                x = result.x
                x = np.where(x > 0, x, 0)

                capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                total_power = np.sum(x * Pmax_val * timeslot_data / self.num_timeslot)

                minimum_capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_outage = np.sum(minimum_capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                minimum_total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data)

                if ((outage == True) and (minimum_outage == False)) == True:
                    capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                    total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data / self.num_timeslot)
                    x = minimum_data / Pmax_val

                end_time = time.time() - start_time

                print('###############################################################')
                print('Trust region IPM (iterative): num_user ' + str(self.total_user) + ' average_demand ' + str(self.average_demand) + '[Mbps]')
                print('data number: ' + str(data) + ', computational time : ' + str(end_time))
                print('Total capacity: ', np.sum(capacity))
                print(capacity)
                print('Energy efficiency: ', EE)
                print('minimum outage occurs: ', minimum_outage)
                print('outage occurs: ', outage)
                print('total power consumption: ', 10.0 * np.log10(total_power))
                print('###############################################################')
                print('\n')

                IPM_opt[data] = x * Pmax_val
                IPM_time[data] = end_time

            # Save for proposed DNN-based timeslot
            np.savetxt('./IPM_data/iterative/power_val/' + str(savefile_str) + '.csv', IPM_opt, delimiter=",")
            np.savetxt('./IPM_data/iterative/execute_time/' + str(savefile_str) + '.csv', IPM_time, delimiter=",")

        if keep_train == 0:
            # Load for proposed DNN-based timeslot
            IPM_opt = np.loadtxt('./IPM_data/iterative/power_val/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)
            IPM_time = np.loadtxt('./IPM_data/iterative/execute_time/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)

        return IPM_opt, IPM_time

    def TR_IPM_PA_algorithm_DQN(self, desired_signal, inter_signal, traffic_demand, channel_gain, real_traffic, timeslot_alloc, minimum_power, keep_train):
        from ipsolver import minimize_constrained, NonlinearConstraint, LinearConstraint, BoxConstraint

        savefile_str = ('Altitude' + str(self.Altitude) + 'active_beam' + str(self.num_active_beam) +
                        'fixed_beam' + str(self.num_fixed_beam) + 'user' + str(self.num_user) + 'timeslot' + str(self.num_timeslot) +
                        'elevation_angle' + str(self.num_angle) + 'Tx_power' + str(self.maximum_transmit_power) +
                        'average_demand' + str(self.average_demand) + 'bandwidth' + str(self.bandwidth)
                        )

        IPM_opt = np.zeros([self.num_data_prop, self.total_user])
        IPM_time = np.zeros(self.num_data_prop)
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)

        timeslot_alloc = np.tile(np.reshape(timeslot_alloc, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])

        power_const = np.eye(self.total_user)

        if keep_train == 1:
            for data in range(self.num_data_prop):
                start_time = time.time()

                desired_data = desired_signal[data]
                inter_data = inter_signal[data]
                demand_data = traffic_demand[data] * self.bandwidth
                channel_data = channel_gain[data]
                real_data = real_traffic[data]
                timeslot_data = timeslot_alloc[data]
                minimum_data = minimum_power[data]

                # x0 = np.zeros([self.total_user])
                x0 = minimum_data / Pmax_val

                fun = lambda x: -np.sum(self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val))
                grad = lambda x: -self.EE_derv(desired_data, inter_data, timeslot_data, x, Pmax_val)

                constraint = self.constraint_linear(desired_data, inter_data, timeslot_data, Pmax_val, demand_data / self.bandwidth)

                linear1 = LinearConstraint(np.ones(self.total_user) * timeslot_data / self.num_timeslot, ("less", 1.0))
                linear2 = LinearConstraint(constraint, ("greater", np.ones(self.total_user) * 1.0))
                # box = BoxConstraint(("interval", 0.0, 1.0))
                linear3 = LinearConstraint(power_const, ("interval", x0, np.ones(self.total_user)))

                result = minimize_constrained(fun, x0, grad, hess='cs', constraints=(linear1, linear2, linear3), method='tr_interior_point',
                                              sparse_jacobian=True, xtol=1e-8, gtol=1e-8, max_iter=100, verbose=0)
                x = result.x
                x = np.where(x > 0, x, 0)

                capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                total_power = np.sum(x * Pmax_val * timeslot_data / self.num_timeslot)

                minimum_capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_outage = np.sum(minimum_capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                minimum_total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data)

                if ((outage == True) and (minimum_outage == False)) == True:
                    capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                    total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data / self.num_timeslot)
                    x = minimum_data / Pmax_val

                end_time = time.time() - start_time

                print('###############################################################')
                print('Trust region IPM (DQN): num_user ' + str(self.total_user) + ' average_demand ' + str(self.average_demand) + '[Mbps]')
                print('data number: ' + str(data) + ', computational time : ' + str(end_time))
                print('Total capacity: ', np.sum(capacity))
                print(capacity)
                print('Energy efficiency: ', EE)
                print('outage occurs: ', outage)
                print('total power consumption: ', 10.0 * np.log10(total_power))
                print('###############################################################')
                print('\n')

                IPM_opt[data] = x * Pmax_val
                IPM_time[data] = end_time

            # Save for proposed DNN-based timeslot
            np.savetxt('./IPM_data/DQN/power_val/' + str(savefile_str) + '.csv', IPM_opt, delimiter=",")
            np.savetxt('./IPM_data/DQN/execute_time/' + str(savefile_str) + '.csv', IPM_time, delimiter=",")

        if keep_train == 0:
            # Load for proposed DNN-based timeslot
            IPM_opt = np.loadtxt('./IPM_data/DQN/power_val/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)
            IPM_time = np.loadtxt('./IPM_data/DQN/execute_time/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)

        return IPM_opt, IPM_time

    def TR_IPM_PA_algorithm_Dueling_DQN(self, desired_signal, inter_signal, traffic_demand, channel_gain, real_traffic, timeslot_alloc, minimum_power, keep_train):
        from ipsolver import minimize_constrained, NonlinearConstraint, LinearConstraint, BoxConstraint

        savefile_str = ('Altitude' + str(self.Altitude) + 'active_beam' + str(self.num_active_beam) +
                        'fixed_beam' + str(self.num_fixed_beam) + 'user' + str(self.num_user) + 'timeslot' + str(self.num_timeslot) +
                        'elevation_angle' + str(self.num_angle) + 'Tx_power' + str(self.maximum_transmit_power) +
                        'average_demand' + str(self.average_demand) + 'bandwidth' + str(self.bandwidth)
                        )

        IPM_opt = np.zeros([self.num_data_prop, self.total_user])
        IPM_time = np.zeros(self.num_data_prop)
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)

        timeslot_alloc = np.tile(np.reshape(timeslot_alloc, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_alloc = np.reshape(timeslot_alloc, [-1, self.num_fixed_beam * self.num_user])

        power_const = np.eye(self.total_user)

        if keep_train == 1:
            for data in range(self.num_data_prop):
                start_time = time.time()

                desired_data = desired_signal[data]
                inter_data = inter_signal[data]
                demand_data = traffic_demand[data] * self.bandwidth
                channel_data = channel_gain[data]
                real_data = real_traffic[data]
                timeslot_data = timeslot_alloc[data]
                minimum_data = minimum_power[data]

                # x0 = np.zeros([self.total_user])
                x0 = minimum_data / Pmax_val

                fun = lambda x: -np.sum(self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val))
                grad = lambda x: -self.EE_derv(desired_data, inter_data, timeslot_data, x, Pmax_val)

                constraint = self.constraint_linear(desired_data, inter_data, timeslot_data, Pmax_val, demand_data / self.bandwidth)

                linear1 = LinearConstraint(np.ones(self.total_user) * timeslot_data / self.num_timeslot, ("less", 1.0))
                linear2 = LinearConstraint(constraint, ("greater", np.ones(self.total_user) * 1.0))
                # box = BoxConstraint(("interval", 0.0, 1.0))
                linear3 = LinearConstraint(power_const, ("interval", x0, np.ones(self.total_user)))

                result = minimize_constrained(fun, x0, grad, hess='cs', constraints=(linear1, linear2, linear3), method='tr_interior_point',
                                              sparse_jacobian=True, xtol=1e-8, gtol=1e-8, max_iter=100, verbose=0)
                x = result.x
                x = np.where(x > 0, x, 0)

                capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, x, Pmax_val)
                outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                total_power = np.sum(x * Pmax_val * timeslot_data / self.num_timeslot)

                minimum_capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                minimum_outage = np.sum(minimum_capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                minimum_total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data)

                if ((outage == True) and (minimum_outage == False)) == True:
                    capacity = self.capacity_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    EE = self.EE_for_IPM(desired_data, inter_data, timeslot_data, minimum_data / Pmax_val, Pmax_val)
                    outage = np.sum(capacity >= real_data * self.bandwidth * 0.999) < self.total_user
                    total_power = np.sum(minimum_data / Pmax_val * Pmax_val * timeslot_data / self.num_timeslot)
                    x = minimum_data / Pmax_val

                end_time = time.time() - start_time

                print('###############################################################')
                print('Trust region IPM (Dueling DQN): num_user ' + str(self.total_user) + ' average_demand ' + str(self.average_demand) + '[Mbps]')
                print('data number: ' + str(data) + ', computational time : ' + str(end_time))
                print('Total capacity: ', np.sum(capacity))
                print(capacity)
                print('Energy efficiency: ', EE)
                print('outage occurs: ', outage)
                print('total power consumption: ', 10.0 * np.log10(total_power))
                print('###############################################################')
                print('\n')

                IPM_opt[data] = x * Pmax_val
                IPM_time[data] = end_time

            # Save for proposed DNN-based timeslot
            np.savetxt('./IPM_data/Dueling_DQN/power_val/' + str(savefile_str) + '.csv', IPM_opt, delimiter=",")
            np.savetxt('./IPM_data/Dueling_DQN/execute_time/' + str(savefile_str) + '.csv', IPM_time, delimiter=",")

        if keep_train == 0:
            # Load for proposed DNN-based timeslot
            IPM_opt = np.loadtxt('./IPM_data/Dueling_DQN/power_val/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)
            IPM_time = np.loadtxt('./IPM_data/Dueling_DQN/execute_time/' + str(savefile_str) + '.csv', delimiter=",", dtype=np.float32)

        return IPM_opt, IPM_time
