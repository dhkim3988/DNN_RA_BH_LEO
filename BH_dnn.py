import tensorflow as tf
import BH_channel
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
import itertools
from collections import deque
import random
import tracemalloc
import matplotlib.pyplot as plt

import numpy as np
import time
import math

tf.keras.backend.set_floatx('float32')

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)

        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

class Timeslot_Allocation_Network(tf.keras.layers.Layer):
    def __init__(self, num_block):
        super(Timeslot_Allocation_Network, self).__init__()
        self.dense1 = Dense(num_block, activation = None, use_bias = True)
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(0.2)

        self.dense2 = Dense(num_block, activation = None, use_bias = True)
        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(0.2)

        self.dense3 = Dense(num_block, activation = None, use_bias = True)
        self.batchnorm3 = BatchNormalization()
        self.dropout3 = Dropout(0.2)

    def call(self, x, batch_prob):
        x = self.dense1(x)
        x = self.batchnorm1(x, training = batch_prob)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training = batch_prob)

        x = self.dense2(x)
        x = self.batchnorm2(x, training = batch_prob)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training = batch_prob)

        x = self.dense3(x)
        x = self.batchnorm3(x, training = batch_prob)
        x = tf.nn.relu(x)
        x = self.dropout3(x, training = batch_prob)

        return x

class Power_Allocation_Coefficient_Network(tf.keras.layers.Layer):
    def __init__(self, num_block):
        super(Power_Allocation_Coefficient_Network, self).__init__()
        self.dense1 = Dense(num_block, activation=None, use_bias=True)
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(0.2)

        self.dense2 = Dense(num_block, activation=None, use_bias=True)
        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(0.2)

        self.dense3 = Dense(num_block, activation=None, use_bias=True)
        self.batchnorm3 = BatchNormalization()
        self.dropout3 = Dropout(0.2)

        self.dense4 = Dense(num_block, activation=None, use_bias=True)
        self.batchnorm4 = BatchNormalization()
        self.dropout4 = Dropout(0.2)


    def call(self, x, batch_prob):
        x = self.batchnorm1(x, training=batch_prob)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        #x = self.dropout1(x, training=batch_prob)

        x = self.batchnorm2(x, training=batch_prob)
        x = self.dense2(x)
        x = tf.nn.relu(x)
        #x = self.dropout2(x, training=batch_prob)

        x = self.batchnorm3(x, training=batch_prob)
        x = self.dense3(x)
        x = tf.nn.relu(x)
        #x = self.dropout3(x, training=batch_prob)

        x = self.batchnorm4(x, training=batch_prob)
        x = self.dense4(x)
        x = tf.nn.relu(x)
        #x = self.dropout4(x, training=batch_prob)

        return x

class DQN_model(Model):
    def __init__(self, num_block):
        super(DQN_model, self).__init__()
        self.num_block = num_block
        self.dense1 = Dense(num_block, activation='sigmoid', use_bias=True)
        self.dense2 = Dense(num_block, activation='sigmoid', use_bias=True)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        return x

class DQN_model2(Model):
    def __init__(self, num_block):
        super(DQN_model2, self).__init__()
        self.num_block = num_block
        self.dense1 = Dense(num_block, activation='relu', use_bias=True)
        self.dense2 = Dense(num_block, activation='relu', use_bias=True)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class DNN_model(Model): # DNN class model
    def __init__(self, Pmax_val, num_timeslot, num_fixed_beam, num_active_beam, num_group, num_user, DL_scheme_val):
        super(DNN_model, self).__init__()
        self.Pmax_val = Pmax_val
        self.num_timeslot = num_timeslot
        self.num_fixed_beam = num_fixed_beam
        self.num_active_beam = num_active_beam
        self.num_group = num_group
        self.num_user = num_user
        self.DL_scheme_val = DL_scheme_val  # set parameter
        self.num_block = 512  # set hidden block

        self.maximum_power_dB = 10.0 * np.log10(self.Pmax_val)

        # network for DL_scheme_val1
        self.dense_timeslot0 = Dense(self.num_group, activation=None, use_bias=True)
        self.timeslot_network = Timeslot_Allocation_Network(self.num_block)

        # network for DL_scheme_val2
        self.dense_power0 = Dense(self.num_fixed_beam*self.num_user, activation=None, use_bias=True)
        self.dense_power1 = Dense(1, activation='sigmoid', use_bias=True)
        self.dense_power2 = Dense(self.num_fixed_beam*self.num_user, activation='sigmoid', use_bias=True)

        self.power_network0 = Power_Allocation_Coefficient_Network(self.num_block)
        self.power_network1 = Power_Allocation_Coefficient_Network(self.num_block)

        # network for DQN-based timeslot allocation (only timeslot)
        self.dense_dqn0 = Dense(self.num_group, activation=None, use_bias=True)
        self.DQN_network = DQN_model(128)

        # network for DQN-based timeslot allocation (reward optimization)
        action_dim = math.perm(self.num_group, 2) + 1
        self.dense_dqn1 = Dense(action_dim, activation=None, use_bias=True)
        self.dense_dqn2 = Dense(1, activation=None, use_bias=True)
        self.DQN_network2 = DQN_model(512)

        self.DQN_network3 = DQN_model2(512)

    def Timeslot_Allocation_NET0(self, X, batch_prob):
        X_timeslot = self.timeslot_network(X, batch_prob)
        X_timeslot = self.dense_timeslot0(X_timeslot)
        X_timeslot = tf.nn.softmax(X_timeslot, -1) * self.num_timeslot
        X_timeslot = tf.where(tf.math.is_finite(X_timeslot), X_timeslot, tf.stop_gradient(tf.constant(1.0)))
        X_timeslot = tf.math.maximum(X_timeslot, tf.stop_gradient(tf.constant(1.0)))
        minimum_timeslot_number = tf.reduce_sum(tf.where(X_timeslot == 1, 1.0, 0.0), -1)
        active_number = self.num_timeslot - minimum_timeslot_number
        sacrifice_number = tf.stop_gradient(tf.math.divide_no_nan(1.0, active_number))
        X_timeslot = tf.where(X_timeslot == 1, 1.0, X_timeslot - tf.expand_dims(sacrifice_number, -1))

        return X_timeslot

    def Power_Allocation_NET0(self, X, batch_prob, timeslot, minimum_power):
        X_power0 = self.power_network0(X, batch_prob)
        X_power1 = self.dense_power0(X_power0)
        X_power2 = self.power_network0(X, batch_prob)
        X_power2 = self.dense_power1(X_power2)
        X_power1 = tf.reshape(X_power1, [-1,self.num_fixed_beam*self.num_user])
        X_power1 = tf.keras.activations.softmax(X_power1, -1)
        X_power = X_power1 * X_power2 * self.Pmax_val / timeslot * self.num_timeslot
        #X_power = tf.math.maximum(minimum_power, X_power)

        X_power = tf.reshape(X_power, [-1,self.num_fixed_beam,self.num_user])

        return X_power

    def Capacity_Estimate_NET0(self, X, batch_prob, capacity_min, capacity_max):
        X_power0 = self.power_network0(X, batch_prob)
        X_power1 = self.dense_power2(X_power0)
        X_power1 = tf.reshape(X_power1, [-1,self.num_fixed_beam*self.num_user])
        X_power = X_power1 * capacity_max
        X_power = tf.math.maximum(capacity_min, X_power)

        return X_power

    def DQN_NET0(self, X):
        X = tf.where(X == 0, tf.stop_gradient(X), X)

        X_timeslot = self.DQN_network(X)
        X_timeslot = self.dense_dqn0(X_timeslot)

        return X_timeslot

    def DQN_NET1(self, X):
        X_timeslot = self.DQN_network2(X)
        X_timeslot = self.dense_dqn1(X_timeslot)

        return X_timeslot

    def DQN_NET2(self, X):
        X_timeslot = self.DQN_network3(X)
        X_timeslot = self.dense_dqn1(X_timeslot)
        X_timeslot2 = self.dense_dqn2(X_timeslot)
        X_timeslot = X_timeslot2 + (X_timeslot - tf.reduce_mean(X_timeslot, axis=-1, keepdims=True))

        return X_timeslot

    def call(self, X, batch_prob=False, timeslot=0, capacity_min=0, capacity_max=0):
        DNN_output = 0

        if self.DL_scheme_val == 1:
            DNN_output = self.Timeslot_Allocation_NET0(X, batch_prob)
        if self.DL_scheme_val == 2:
            DNN_output = self.DQN_NET1(X)
        if self.DL_scheme_val == 3:
            DNN_output = self.DQN_NET2(X)
        if self.DL_scheme_val == 4:
            DNN_output = self.Power_Allocation_NET0(X, batch_prob, timeslot, capacity_min)
        if self.DL_scheme_val == 5:
            DNN_output = self.Capacity_Estimate_NET0(X, batch_prob, capacity_min, capacity_max)
        if self.DL_scheme_val == 6:
            DNN_output = self.DQN_NET2(X)

        return DNN_output

    def DQN_action(self, X):
        X_timeslot = self.DQN_network3(X)
        X_timeslot = self.dense_dqn1(X_timeslot)

        return X_timeslot

class Beamhopping_train:
    def __init__(self, Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                 dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, num_instances, seed, learning_rate, training_epochs, batch_size, lambda_val, DL_scheme_val, ckpt, reuse):
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

        self.learning_rate = learning_rate[DL_scheme_val]
        self.training_epochs = training_epochs[DL_scheme_val]
        self.batch_size = batch_size[DL_scheme_val]
        self.lambda_val = lambda_val
        self.DL_scheme_val = DL_scheme_val

        self.ckpt = ckpt
        self.reuse = reuse

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

    def outage_rate(self, capacity, traffic_demand, test_num_data_prop):
        capacity = np.reshape(capacity, [-1, self.total_user])
        traffic_demand = np.reshape(traffic_demand, [-1, self.total_user])
        outage = np.sum(np.where(capacity >= (traffic_demand * 0.99), 1, 0), axis=-1)
        outage = np.where(outage >= self.total_user, 0, 1)
        outage = np.sum(outage) / test_num_data_prop * 100.0

        return outage

    def Normalized_input(self, group_channel_gain_val, snapshot_index, group_snapshot_index, traffic_demand_data, timeslot_data):
        traffic_demand = np.expand_dims(traffic_demand_data, (1, 3)) * np.expand_dims(snapshot_index, (1, 4)) * np.expand_dims(group_snapshot_index, (2, 4))
        input_traffic_demand = np.reshape(np.sum(traffic_demand, (1, 3)), [-1, self.num_fixed_beam * self.num_user]) / self.bandwidth

        input_channel = np.sum(group_channel_gain_val * np.expand_dims(group_snapshot_index, (2, 4)) * np.expand_dims(group_snapshot_index, (3, 4)), 1)
        input_channel = np.reshape(np.sum(np.reshape(input_channel, [-1, self.num_fixed_beam, self.num_group, self.num_active_beam, self.num_user]), 2), [-1, self.num_group, self.num_active_beam ** 2, self.num_user])
        input_channel = np.reshape(np.swapaxes(input_channel, 2, 3), [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam])

        input_traffic_rate = np.swapaxes(np.reshape(input_traffic_demand, [-1, self.num_group, self.num_active_beam, self.num_user]), 2, 3)
        input_traffic_rate = np.expand_dims(input_traffic_rate, 3) * np.eye(self.num_active_beam)
        input_traffic_value = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.expand_dims(timeslot_data, (2, 3, 4))) - 1.0

        input_desired = np.where(input_traffic_rate == 0, 0.0, input_channel / input_traffic_value)
        input_desired = np.sum(input_desired, axis=3)
        input_desired2 = np.where(input_traffic_rate == 0, 0.0, input_channel)
        input_desired2 = np.sum(input_desired2, axis=3)
        input_inter = np.where(input_traffic_rate == 0, input_channel, 0.0)
        input_inter = np.sum(input_inter, axis=3)
        input_state1 = np.reshape(np.log10(1.0 + input_desired / (input_inter + 1.0)), [-1, self.num_group * self.num_user * self.num_active_beam])
        input_state2 = np.reshape(np.log10(1.0 + input_desired), [-1, self.num_group * self.num_user * self.num_active_beam])
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        equal_power = Pmax_val / self.total_user / (self.num_timeslot ** 2.0)
        input_state3 = np.reshape(np.log10(1.0 + (input_inter * equal_power + 1.0) / input_desired), [-1, self.num_group * self.num_user * self.num_active_beam])

        DNN_input1 = np.concatenate((input_state1, timeslot_data / (self.num_timeslot / self.num_group)), axis=-1)
        DNN_input2 = np.concatenate((input_state2, timeslot_data / (self.num_timeslot / self.num_group)), axis=-1)
        DNN_input3 = np.concatenate((input_state3, timeslot_data / (self.num_timeslot / self.num_group)), axis=-1)

        DNN_input3 = input_state3

        return DNN_input1, DNN_input2, DNN_input3

    def capacity_min_max(self, input_traffic_rate, input_channel, timeslot_data, capacity_min, test_num_data_prop):
        capacity_min = np.reshape(capacity_min, [-1, self.total_user])
        current_rate = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / np.reshape(timeslot_data, [test_num_data_prop, self.num_group, 1, 1, 1])) - 1.0
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
        addi_val = np.where(addi_index, (desired / inter_sum) * 0.99, current_rate)
        addi_val = np.where(addi_val > current_rate, current_rate, addi_val)
        addi_val = np.swapaxes(np.sum(addi_val, -1), 2, 3)
        addi_val = np.reshape(addi_val, [-1, self.total_user])
        desired = np.where(input_traffic_rate == 0, 0.0, input_channel)
        inter_sum = np.sum(input_channel - desired, axis=4, keepdims=True)
        capacity_max = np.log2((desired / inter_sum) * 0.99 + 1.0)
        capacity_max = np.reshape(np.swapaxes(np.sum(capacity_max, -1), 2, 3), [-1, self.total_user]) / self.num_user / self.num_timeslot * timeslot_for_state
        capacity_min = capacity_min / self.bandwidth

        return capacity_min, capacity_max

    def matrix_inversion_based_power_allocation(self, input_traffic_rate, input_channel, timeslot_data):
        current_rate = 2.0 ** (input_traffic_rate * self.num_user * self.num_timeslot / tf.reshape(timeslot_data, [-1, self.num_group, 1, 1, 1])) - 1.0
        channel_state = tf.where(input_traffic_rate == 0, -input_channel, input_channel / current_rate)
        power_matrix = tf.linalg.inv(tf.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        eig_val = tf.math.real(tf.linalg.eigvals(tf.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam])))
        eig_index = tf.tile(tf.reduce_sum(tf.where(eig_val < 0.0, 1.0, 0.0), axis=-1, keepdims=True), [1, 1, 1, self.num_active_beam]) > 0.0
        eig_index = tf.reshape(tf.experimental.numpy.swapaxes(eig_index, 2, 3), [-1, self.num_fixed_beam, self.num_user])
        power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 2, 3)
        equal_power = tf.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])

        additional_val = tf.math.minimum(tf.reduce_sum(channel_state, 4), 0.0) * -1.0
        additional_val = tf.expand_dims(additional_val, 4)
        addi_index = tf.tile(additional_val, [1, 1, 1, 1, self.num_active_beam]) > 0.0
        desired = tf.where(input_traffic_rate == 0, 0.0, input_channel)
        inter_sum = tf.reduce_sum(input_channel - desired, axis=4, keepdims=True)
        addi_val = tf.where(addi_index, (desired / inter_sum) * 0.99, current_rate)
        addi_val = tf.where(addi_val > current_rate, current_rate, addi_val)
        channel_state = tf.where(input_traffic_rate == 0, -input_channel, input_channel / addi_val)
        power_matrix = tf.linalg.inv(tf.reshape(channel_state, [-1, self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
        power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 2, 3)
        eqp = tf.reshape(power_value, [-1, self.num_fixed_beam, self.num_user])
        equal_power = tf.where(eig_index, eqp, equal_power)

        matrix_inverse_based_power_allocation = equal_power

        return matrix_inverse_based_power_allocation

    @tf.function
    def Unsupervised_Timeslot_Allocation_train_step(self, batch_desired, batch_inter, batch_demand, batch_channel, batch_snapshot, batch_group_snapshot, optimizer, model, train_loss, train_capacity, batch_prob, lam, Pmax_val):
        with (tf.GradientTape() as tape):
            desired_concat = tf.expand_dims(batch_desired, -1)
            inter_concat = batch_inter
            demand_concat = tf.expand_dims(batch_demand, -1)
            DNN_input = tf.reshape(tf.concat([desired_concat, inter_concat, demand_concat], -1), [-1,self.num_fixed_beam*self.num_user*(1+self.num_fixed_beam*self.num_user+1)])
            DNN_timeslot = model(DNN_input, batch_prob)

            timeslot_for_loss = tf.tile(tf.reshape(DNN_timeslot, [-1,self.num_group,1,1,1]), [1,1,self.num_fixed_beam,self.num_fixed_beam,self.num_user]) \
                                * tf.reshape(batch_snapshot, [-1,1,self.num_fixed_beam,self.num_fixed_beam,1]) * tf.reshape(batch_group_snapshot, [-1, self.num_group, 1, self.num_fixed_beam, 1])
            timeslot_for_loss = tf.reshape(tf.reduce_sum(timeslot_for_loss, [1,3]), [-1,self.num_fixed_beam*self.num_user])

            equal_power = Pmax_val / self.total_user / (self.num_timeslot / self.num_group)

            desired_power = batch_desired * equal_power
            inter_power = tf.reduce_sum(batch_inter * equal_power, -1) + 1.0

            traffic_for_loss = 2.0 ** (self.bandwidth * batch_demand * self.num_timeslot / (self.bandwidth / self.num_user) / timeslot_for_loss) - 1.0
            loss = tf.reduce_mean(tf.reduce_sum(timeslot_for_loss * traffic_for_loss * inter_power / batch_desired, -1))
            capacity = tf.reduce_mean(tf.reduce_sum(self.bandwidth / self.num_user * timeslot_for_loss / self.num_timeslot * tf.math.log(1.0 + desired_power / inter_power) / tf.math.log(2.0), -1))

            #capacity = self.data_rate(batch_channel, equal_power, batch_snapshot, batch_group_snapshot, DNN_timeslot)
            #loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(batch_demand - capacity), [1,2]))
            #loss = self.closed_form_PAC(batch_channel, equal_power, batch_snapshot, batch_group_snapshot, DNN_timeslot, batch_demand)
            #loss = tf.expand_dims(loss, 1) * tf.reshape(DNN_timeslot, [-1,self.num_group,1,1]) * tf.expand_dims(batch_group_snapshot, -1)
            #loss = tf.reduce_mean(tf.reduce_sum(loss / Pmax_val, [1,2,3]))

        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_capacity(capacity)

    def Unsupervised_Timeslot_Allocation_train(self, input_desired, input_inter, input_demand, channel_gain, snapshot_index, group_snapshot_index, traffic_demand):
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        batch_prob = True
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        channel_gain = tf.cast(channel_gain, tf.float32)
        snapshot_index = tf.cast(snapshot_index, tf.float32)
        group_snapshot_index = tf.cast(group_snapshot_index, tf.float32)
        traffic_demand = tf.cast(traffic_demand, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_loss = tf.keras.metrics.Mean()
        train_capacity = tf.keras.metrics.Mean()
        lam = self.lambda_val[self.DL_scheme_val]

        train_ds = tf.data.Dataset.from_tensor_slices((input_desired, input_inter, input_demand, channel_gain,
                                                       snapshot_index, group_snapshot_index)).batch(self.batch_size)

        if self.reuse == 1:
            model.load_weights(self.ckpt[self.DL_scheme_val])  # restore checkpoint file
        for epoch in range(self.training_epochs):
            start_time = time.time()
            for batch_desired, batch_inter, batch_demand, batch_channel, batch_snapshot, batch_group_snapshot in train_ds:
                self.Unsupervised_Timeslot_Allocation_train_step(batch_desired, batch_inter, batch_demand, batch_channel, batch_snapshot, batch_group_snapshot, optimizer, model, train_loss, train_capacity, batch_prob, lam, Pmax_val)
            total_time = time.time() - start_time
            template = 'Unsupervised Timeslot Allocation Training: \n epoch : {}, loss : {:.3f}, capacity : {:.3f} [Mbps], time : {:.3f}'
            print(template.format(epoch + 1, train_loss.result(), train_capacity.result(), total_time))

        model.save_weights(self.ckpt[self.DL_scheme_val])

        return 0

    #@tf.function
    def Unsupervised_Power_Allocation_train_step(self, batch_desired, batch_inter, batch_demand, batch_input, batch_timeslot, batch_minimum_power, optimizer, model, train_loss, train_capacity, train_EE,
                                                 train_outage, train_power_consumption, batch_prob, lam, Pmax_val):
        with (tf.GradientTape() as tape):
            timeslot_for_state = tf.tile(tf.reshape(batch_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_for_state = tf.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
            DNN_power = model(batch_input, batch_prob, timeslot_for_state, batch_minimum_power)
            DNN_power = tf.reshape(DNN_power, [-1, self.num_fixed_beam * self.num_user])
            desired_SNR = batch_desired * DNN_power
            inter_SNR = tf.reduce_sum(batch_inter * tf.expand_dims(DNN_power, 1), -1) + 1.0
            capacity = self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0)
            capacity_gap = tf.nn.relu(self.bandwidth * batch_demand - capacity)
            capacity_loss = tf.nn.relu(capacity - self.bandwidth * batch_demand)

            total_power = tf.reduce_sum(timeslot_for_state * DNN_power, -1)
            #print(total_power)

            loss = tf.reduce_mean(tf.reduce_sum(-lam * capacity_loss, -1) / (total_power / 1000.0 + 1.0) * self.num_timeslot + tf.reduce_sum(capacity_gap, -1))
            #loss = tf.reduce_sum(-lam * capacity_loss + capacity_gap, -1)

            outage = tf.reduce_sum(tf.where(capacity >= (self.bandwidth * batch_demand * 0.999), 1, 0), axis=-1)
            outage = tf.where(outage >= self.total_user, 0, 1)
            outage = tf.reduce_sum(outage)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_capacity(tf.reduce_sum(capacity, -1))
        train_EE(tf.reduce_sum(capacity, -1) / (total_power / 1000.0 + 1.0) * self.num_timeslot)
        train_outage(outage)
        train_power_consumption(tf.reduce_mean(total_power / self.num_timeslot))

    def Unsupervised_Power_Allocation_train(self, input_desired, input_inter, input_demand, input_channel, input_traffic_rate, input_timeslot, DNN_input):
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        batch_prob = True
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        lam = self.lambda_val[self.DL_scheme_val]
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        DNN_input = tf.cast(DNN_input, tf.float32)

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)

        if self.reuse == 1:
            model.load_weights(self.ckpt[self.DL_scheme_val])  # restore checkpoint file

        timeslot_for_state = tf.tile(tf.reshape(input_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_for_state = tf.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
        minimum_power = self.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, input_timeslot)
        minimum_power = tf.reshape(minimum_power, [-1, self.num_fixed_beam * self.num_user])
        desired_SNR = input_desired * minimum_power
        inter_SNR = tf.reduce_sum(input_inter * tf.expand_dims(minimum_power, 1), -1) + 1.0
        minimum_capacity = self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0)

        minimum_outage = tf.reduce_sum(tf.where(minimum_capacity >= (self.bandwidth * input_demand * 0.999), 1, 0), axis=-1)
        minimum_outage = tf.where(minimum_outage >= self.total_user, 0, 1)
        minimum_outage = tf.reduce_sum(minimum_outage) / self.num_data_prop * 100.0

        minimum_total_power = tf.reduce_sum(minimum_power * timeslot_for_state, -1)
        minimum_EE = tf.reduce_mean(tf.reduce_sum(minimum_capacity, -1) / (minimum_total_power / 1000.0 + 1.0) * self.num_timeslot)
        minimum_total_power = 10.0 * tf.math.log(tf.reduce_mean(minimum_total_power) / self.num_timeslot) / tf.math.log(10.0)
        minimum_capacity = tf.reduce_mean(tf.reduce_sum(minimum_capacity, -1))

        train_ds = tf.data.Dataset.from_tensor_slices((input_desired, input_inter, input_demand, DNN_input, input_timeslot, minimum_power)).batch(self.batch_size)

        for epoch in range(self.training_epochs):
            train_loss = tf.keras.metrics.Mean()
            train_capacity = tf.keras.metrics.Mean()
            train_outage = tf.keras.metrics.Sum()
            train_power_consumption = tf.keras.metrics.Mean()
            train_EE = tf.keras.metrics.Mean()

            start_time = time.time()
            for batch_desired, batch_inter, batch_demand, batch_input, batch_timeslot, batch_minimum_power in train_ds:
                self.Unsupervised_Power_Allocation_train_step(batch_desired, batch_inter, batch_demand, batch_input, batch_timeslot, batch_minimum_power, optimizer, model, train_loss, train_capacity, train_EE,
                                                              train_outage, train_power_consumption, batch_prob, lam, Pmax_val)
            total_time = time.time() - start_time

            template2 = 'Power minimization performance \n capacity: {:.3f} [Mbps], EE: {:.3f} [Mbps/J], outage rate : {:.3f}%, power consumption : {:.3f} [dBm]'
            template = 'Unsupervised learning-based power allocation: \n epoch : {}, loss : {:.3f}, capacity: {:.3f} [Mbps], EE: {:.3f} [Mbps/J], outage rate : {:.3f}%, power consumption : {:.3f} [dBm], time : {:.3f}'
            print(template2.format(minimum_capacity, minimum_EE, minimum_outage, minimum_total_power))
            print(template.format(epoch + 1, train_loss.result(), train_capacity.result(), train_EE.result(), train_outage.result() / self.num_data_prop * 100.0, 10.0 * np.log10(train_power_consumption.result()), total_time))
            print('\n')

        model.save_weights(self.ckpt[self.DL_scheme_val])

        return 0

    #@tf.function
    def Unsupervised_Capacity_Estimation_train_step(self, batch_desired, batch_inter, batch_demand, batch_input, batch_timeslot, batch_channel, optimizer, model, train_loss, train_capacity, train_EE,
                                                 train_outage, train_power_consumption, batch_prob, lam, capacity_min, capacity_max):
        with (tf.GradientTape() as tape):
            timeslot_for_state = tf.tile(tf.reshape(batch_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
            timeslot_for_state = tf.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
            DNN_capacity = model(batch_input, batch_prob, timeslot_for_state, capacity_min, capacity_max)
            DNN_capacity = tf.reshape(DNN_capacity, [-1, self.num_fixed_beam * self.num_user])

            input_traffic_rate = tf.experimental.numpy.swapaxes(tf.reshape(DNN_capacity, [-1, self.num_group, self.num_active_beam, self.num_user]), 2, 3)
            #input_traffic_rate = tf.transpose(input_traffic_rate, perm=[0,1,3,2])
            input_traffic_rate = tf.expand_dims(input_traffic_rate, 3) * tf.eye(self.num_active_beam)
            input_traffic_rate = tf.where(input_traffic_rate == 0, tf.stop_gradient(input_traffic_rate), input_traffic_rate)

            DNN_power = self.matrix_inversion_based_power_allocation(input_traffic_rate, batch_channel, batch_timeslot)
            DNN_power = tf.reshape(DNN_power, [-1, self.num_fixed_beam * self.num_user])
            DNN_power = tf.where(tf.math.is_finite(DNN_power), DNN_power, 0.0)

            desired_SNR = batch_desired * DNN_power
            inter_SNR = tf.reduce_sum(batch_inter * tf.expand_dims(DNN_power, 1), -1) + 1.0
            capacity = self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0)
            #capacity_gap = tf.nn.relu(self.bandwidth * batch_demand - capacity)
            #capacity_loss = tf.nn.relu(capacity - self.bandwidth * batch_demand)

            outage = tf.reduce_sum(tf.where(capacity >= (self.bandwidth * batch_demand * 0.999), 1, 0), axis=-1)
            outage = tf.where(outage >= self.total_user, 0, 1)
            outage = tf.reduce_sum(outage)

            total_power = tf.reduce_sum(timeslot_for_state * DNN_power, -1)

            #loss = (1.0 + total_power/1000.0) - lam * tf.reduce_sum(DNN_capacity * self.bandwidth, -1)
            loss = tf.reduce_mean(-tf.reduce_sum(DNN_capacity * self.bandwidth / self.num_user * timeslot_for_state, -1) / (total_power / 1000.0 + 1.0) + lam * tf.nn.relu(total_power / self.num_timeslot - 10.0 ** (self.maximum_transmit_power / 10.0)))
            #loss = tf.reduce_mean(-tf.reduce_sum(DNN_capacity, -1) + lam * tf.nn.relu(total_power - 10.0 ** (self.maximum_transmit_power / 10.0)))
            #loss = tf.reduce_mean(DNN_power)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_capacity(tf.reduce_sum(capacity, -1))
        train_EE(tf.reduce_sum(capacity, -1) / (total_power / 1000.0 + 1.0) * self.num_timeslot)
        train_outage(outage)
        train_power_consumption(total_power / self.num_timeslot)

    def Unsupervised_Capacity_Estimation_train(self, input_desired, input_inter, input_demand, input_channel, input_traffic_rate, input_timeslot, DNN_input, capacity_min, capacity_max):
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        batch_prob = True
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        lam = self.lambda_val[self.DL_scheme_val]
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        capacity_min = tf.cast(capacity_min, tf.float32)
        capacity_max = tf.cast(capacity_max, tf.float32)

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)

        if self.reuse == 1:
            model.load_weights(self.ckpt[self.DL_scheme_val])  # restore checkpoint file

        timeslot_for_state = tf.tile(tf.reshape(input_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_for_state = tf.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
        minimum_power = self.matrix_inversion_based_power_allocation(input_traffic_rate, input_channel, input_timeslot)
        minimum_power = tf.reshape(minimum_power, [-1, self.num_fixed_beam * self.num_user])
        desired_SNR = input_desired * minimum_power
        inter_SNR = tf.reduce_sum(input_inter * tf.expand_dims(minimum_power, 1), -1) + 1.0
        minimum_capacity = self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0)

        minimum_outage = tf.reduce_sum(tf.where(minimum_capacity >= (self.bandwidth * input_demand * 0.999), 1, 0), axis=-1)
        minimum_outage = tf.where(minimum_outage >= self.total_user, 0, 1)
        minimum_outage = tf.reduce_sum(minimum_outage) / self.num_data_prop * 100.0

        minimum_total_power = tf.reduce_sum(minimum_power * timeslot_for_state, -1)
        minimum_EE = tf.reduce_mean(tf.reduce_sum(minimum_capacity, -1) / (minimum_total_power / 1000.0 + 1.0) * self.num_timeslot)
        minimum_total_power = tf.reduce_mean(10.0 * tf.math.log(minimum_total_power / self.num_timeslot) / tf.math.log(10.0))
        minimum_capacity = tf.reduce_mean(tf.reduce_sum(minimum_capacity, -1))

        train_ds = tf.data.Dataset.from_tensor_slices((input_desired, input_inter, input_demand, DNN_input, input_timeslot, input_channel, capacity_min, capacity_max)).batch(self.batch_size).shuffle(buffer_size=self.num_data_prop + 1)

        for epoch in range(self.training_epochs):
            train_loss = tf.keras.metrics.Mean()
            train_capacity = tf.keras.metrics.Mean()
            train_outage = tf.keras.metrics.Sum()
            train_power_consumption = tf.keras.metrics.Mean()
            train_EE = tf.keras.metrics.Mean()

            start_time = time.time()
            for batch_desired, batch_inter, batch_demand, batch_input, batch_timeslot, batch_channel, batch_capacity_min, batch_capacity_max in train_ds:
                self.Unsupervised_Capacity_Estimation_train_step(batch_desired, batch_inter, batch_demand, batch_input, batch_timeslot, batch_channel, optimizer, model, train_loss, train_capacity, train_EE,
                                                                  train_outage, train_power_consumption, batch_prob, lam, batch_capacity_min, batch_capacity_max)
            total_time = time.time() - start_time

            template2 = 'Power minimization performance \n capacity: {:.3f} [Mbps], EE: {:.3f} [Mbps/J], outage rate : {:.3f}%, power consumption : {:.3f} [dBm]'
            template = 'Unsupervised capacity estimation: \n epoch : {}, loss : {:.3f}, capacity: {:.3f} [Mbps], EE: {:.3f} [Mbps/J], outage rate : {:.3f}%, power consumption : {:.3f} [dBm], time : {:.3f}'
            print(template2.format(minimum_capacity, minimum_EE, minimum_outage, minimum_total_power))
            print(template.format(epoch + 1, train_loss.result(), train_capacity.result(), train_EE.result(), train_outage.result() / self.num_data_prop * 100.0, 10.0 * np.log10(train_power_consumption.result()), total_time))
            print('\n')

        model.save_weights(self.ckpt[self.DL_scheme_val])

        return 0

    def target_update(self, model, target_model):
        weights = model.get_weights()
        target_model.set_weights(weights)

    def replay(self, buffer, state_dim, model, target_model, discount_factor, optimizer, train_loss):
        states, actions, rewards, next_states, done = buffer.sample(self.batch_size)
        states = np.reshape(states, [-1, state_dim])
        next_states = np.reshape(next_states, [-1, state_dim])
        next_q_values = np.max(target_model(next_states), axis=-1)

        targets = rewards + (1 - done) * next_q_values * discount_factor

        self.Single_DQN_Timeslot_Allocation_train_step(states, actions, targets, model, optimizer, train_loss)

    def get_action(self, state, state_dim, epsilon, model):
        state = np.reshape(state, [1, state_dim])

        if np.random.random() < epsilon:
            random_action = np.zeros([self.num_group])
            randint = np.random.randint(self.num_group)
            random_action[randint] = 1

            return random_action

        else:
            q_value = model(state)[0]
            max_q = np.max(q_value, keepdims=True)
            action_index = q_value == max_q

            return action_index

    #@tf.function
    def Single_DQN_Timeslot_Allocation_train_step(self, states, actions, targets, model, optimizer, train_loss):
        with (tf.GradientTape() as tape):
            DNN_timeslot = model(states)
            DNN_timeslot = DNN_timeslot
            targets = tf.where(tf.cast(actions, tf.float32) == 1.0, tf.expand_dims(tf.cast(targets, tf.float32), -1), DNN_timeslot)
            loss_function = tf.keras.losses.LogCosh()
            loss = loss_function(DNN_timeslot, targets)
            #print(DNN_timeslot)
            #loss = tf.square(DNN_timeslot - tf.cast(targets, tf.float32))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    def Single_DQN_Timeslot_Allocation_train(self, input_desired, input_inter, input_demand, epsilon, eps_decay, eps_min, discount_factor, input_channel, input_traffic_rate, input_value):
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        batch_prob = True
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        lam = self.lambda_val[self.DL_scheme_val]
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        input_value = tf.cast(input_value, tf.float32)

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        target_model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)

        len_capacity = 1000
        buffer = ReplayBuffer(len_capacity)

        max_episodes = self.num_timeslot

        if self.reuse == 1:
            model.load_weights(self.ckpt[self.DL_scheme_val])  # restore checkpoint file
            target_model.load_weights(self.ckpt[self.DL_scheme_val])
            epsilon = eps_min

        if self.reuse == 0:
            self.target_update(model, target_model)

        # state_dim = int(self.num_group*self.num_user*self.num_active_beam*self.num_active_beam + self.num_group)
        state_dim = int(self.num_group * self.num_user * self.num_active_beam)
        action_dim = int(math.perm(self.num_group, 2) + 1)

        action_value_0 = np.zeros(self.num_group)
        action_value_sample = action_value_0 + 0
        action_value_sample[0] = 1
        action_value_sample[-1] = -1

        action_value = np.array(list(itertools.permutations(action_value_sample, self.num_group)))
        action_value = np.unique(action_value, axis=0)
        action_value = np.concatenate([np.expand_dims(action_value_0, 0), action_value], axis=0)
        action_value = tf.cast(action_value, tf.float32)

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot

        zero_action = np.zeros(action_dim)
        zero_action[0] = 1

        eps_count = 0

        for epoch in range(self.num_data_prop):
            train_loss = tf.keras.metrics.Mean()
            train_capacity = tf.keras.metrics.Mean()
            train_reward = tf.keras.metrics.Sum()
            train_power_consumption = tf.keras.metrics.Mean()

            current_timeslot = tf.cast(equal_timeslot + 0.0, tf.float32)
            desired_instance = input_desired[epoch]
            inter_instance = input_inter[epoch]
            demand_instance = input_demand[epoch]
            channel_instance = input_channel[epoch]
            traffic_rate_instance = input_traffic_rate[epoch]
            input_instance = tf.reshape(input_value[epoch], [self.num_group * self.num_user * self.num_active_beam * self.num_active_beam])

            current_rate = 2.0 ** (traffic_rate_instance * self.num_user / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0
            channel_state = tf.where(traffic_rate_instance == 0, -channel_instance, channel_instance / current_rate)
            channel_state = tf.reshape(channel_state, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam)

            state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance / current_rate)
            state_desired = tf.reduce_sum(state_desired, axis=2)
            state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
            state_inter = tf.reduce_sum(state_inter, axis=2)
            current_state = state_desired / state_inter
            current_state = tf.reshape(current_state, state_dim)

            power_matrix = tf.linalg.inv(tf.reshape(channel_state, [self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
            power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 1, 2)
            power_value = tf.reshape(power_value, [self.num_fixed_beam * self.num_user])
            power_value = tf.where(power_value < 0, Pmax_val, power_value)

            timeslot_for_state = tf.tile(tf.reshape(current_timeslot, [self.num_group, 1, 1]), [1, self.num_active_beam, self.num_user])
            timeslot_for_state = tf.reshape(timeslot_for_state, [self.num_fixed_beam * self.num_user])

            traffic_rate = 2.0 ** (demand_instance * self.num_user * self.num_timeslot / timeslot_for_state) - 1.0
            equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user
            desired_SNR = desired_instance
            inter_SNR = tf.reduce_sum(inter_instance * equal_power, -1) + 1.0

            current_reward = 10.0 * np.log10(tf.reduce_sum(timeslot_for_state / self.num_timeslot * traffic_rate * inter_SNR / desired_SNR, -1))

            start_time = time.time()
            done = 0
            for ep in range(max_episodes):
                current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0
                # current_state = tf.concat([input_instance, current_timeslot / (self.num_timeslot / self.num_group)], -1)

                state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance / current_rate)
                state_desired = tf.reduce_sum(state_desired, axis=2)
                state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
                state_inter = tf.reduce_sum(state_inter, axis=2)
                current_state = state_desired / state_inter
                current_state = tf.reshape(tf.math.log(1.0 + current_state) / tf.math.log(10.0), state_dim)
                current_state = tf.concat([current_state, current_timeslot / (self.num_timeslot / self.num_group)], -1)

                current_action = self.get_action2(current_state, state_dim + self.num_group, action_dim, epsilon, model)

                while True:
                    cal_timeslot = current_timeslot + tf.reduce_sum(tf.expand_dims(tf.cast(current_action, tf.float32), -1) * action_value, 0)
                    if any(cal_timeslot == 0) == True:
                        random_action = np.zeros([action_dim])
                        randint = np.random.randint(action_dim)
                        random_action[randint] = 1
                        current_action = random_action

                    if any(cal_timeslot == 0) != True:
                        break

                # if (ep == (max_episodes - 1)) or (np.sum(current_action == zero_action) == action_dim):
                #    done = 1
                if np.sum(current_action == zero_action) == action_dim:
                    done = 1

                current_timeslot = tf.where(any(cal_timeslot == 0), current_timeslot, cal_timeslot)
                current_action = tf.where(any(cal_timeslot == 0), zero_action, current_action)
                current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0

                channel_state = tf.where(traffic_rate_instance == 0, -channel_instance, channel_instance / current_rate)
                channel_state = tf.reshape(channel_state, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam)
                power_matrix = tf.linalg.inv(tf.reshape(channel_state, [self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
                power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 1, 2)
                power_value = tf.reshape(power_value, [self.num_fixed_beam * self.num_user])
                equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.num_fixed_beam / self.num_user / timeslot_for_state
                power_value = tf.where(power_value < 0, equal_power, power_value)
                power_value = tf.where(np.sum(power_value) > Pmax_val, power_value / np.sum(power_value) * Pmax_val / timeslot_for_state, power_value)

                timeslot_for_state = tf.tile(tf.reshape(current_timeslot, [self.num_group, 1, 1]), [1, self.num_active_beam, self.num_user])
                timeslot_for_state = tf.reshape(timeslot_for_state, [self.num_fixed_beam * self.num_user])

                inter_SNR = tf.reduce_sum(inter_instance * power_value, -1) + 1.0

                desired_SNR = desired_instance * power_value
                capacity = tf.reduce_sum(self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0), -1)
                capacity_gap = tf.math.abs(self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0) - self.bandwidth * demand_instance)

                state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance / current_rate)
                state_desired = tf.reduce_sum(state_desired, axis=2)
                state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
                state_inter = tf.reduce_sum(state_inter, axis=2)
                next_state = state_desired / state_inter
                next_state = tf.reshape(tf.math.log(1.0 + next_state) / tf.math.log(10.0), state_dim)
                next_state = tf.concat([next_state, current_timeslot / (self.num_timeslot / self.num_group)], -1)

                previous_reward = current_reward

                traffic_rate = 2.0 ** (demand_instance * self.num_user / timeslot_for_state) - 1.0
                equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.num_fixed_beam / self.num_user / timeslot_for_state
                desired_SNR = desired_instance
                inter_SNR = tf.reduce_sum(inter_instance * equal_power, -1) + 1.0

                current_reward = 10.0 * np.log10(tf.reduce_sum(timeslot_for_state * traffic_rate * inter_SNR / desired_SNR, -1))

                put_reward = (previous_reward / current_reward) - 1.0
                put_reward = tf.math.sign(put_reward)

                buffer.put(current_state, current_action, put_reward, next_state, done)
                train_capacity(tf.reduce_sum(capacity_gap))
                train_reward(put_reward)
                train_power_consumption(current_reward)

                if buffer.size() >= self.batch_size:
                    self.replay2(buffer, state_dim + self.num_group, model, target_model, discount_factor, optimizer, train_loss)

                if done == 1:
                    break

            total_time = time.time() - start_time
            print(current_timeslot)

            epsilon = np.maximum(eps_min, epsilon * eps_decay)

            if epoch % 10 == 0 and epoch > 0 and buffer.size() >= self.batch_size:
                self.target_update(model, target_model)
                model.save_weights(self.ckpt[self.DL_scheme_val])

            template = 'DQN-based timeslot training (only timeslot): \n epoch : {}, loss : {:.3f}, gap : {:.3f} [Mbps], cumulative reward : {:.3f}, power consumption : {:.3f} [dBm], time : {:.3f}'
            print(template.format(epoch + 1, train_loss.result(), train_capacity.result(), train_reward.result(), 10.0 * np.log10(train_power_consumption.result()), total_time))

        model.save_weights(self.ckpt[self.DL_scheme_val])

        return 0

    def Single_DQN_Timeslot_Allocation_graph(self, input_desired, input_inter, input_demand, epsilon, eps_decay, eps_min, discount_factor, input_channel, input_traffic_rate, input_value,
                                             test_input_channel, test_input_traffic_rate, test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, test_input_traffic_demand, test_num_data_prop):
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        input_value = tf.cast(input_value, tf.float32)
        tf_input_channel = tf.cast(test_input_channel, tf.float32)
        tf_input_traffic_rate = tf.cast(test_input_traffic_rate, tf.float32)
        tf_group_channel_gain = tf.cast(test_group_channel_gain, tf.float32)
        tf_snapshot_index = tf.cast(test_snapshot_index, tf.float32)
        tf_group_snapshot_index = tf.cast(test_group_snapshot_index, tf.float32)
        tf_input_traffic_demand = tf.cast(test_input_traffic_demand, tf.float32)

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        target_model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)

        len_capacity = 1000
        buffer = ReplayBuffer(len_capacity)

        max_episodes = self.num_timeslot - self.num_group

        self.target_update(model, target_model)

        state_dim = int(self.num_group * self.num_user * self.num_active_beam)
        action_dim = int(math.perm(self.num_group, 2) + 1)

        action_value_0 = np.zeros(self.num_group)
        action_value_sample = action_value_0 + 0
        action_value_sample[0] = 1
        action_value_sample[-1] = -1

        action_value = np.array(list(itertools.permutations(action_value_sample, self.num_group)))
        action_value = np.unique(action_value, axis=0)
        action_value = np.concatenate([np.expand_dims(action_value_0, 0), action_value], axis=0)
        action_value = tf.cast(action_value, tf.float32)

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot

        zero_action = np.zeros(action_dim)
        zero_action[0] = 1

        training_epochs = self.num_data_prop

        load_EE = np.zeros([training_epochs])
        load_outage = np.zeros([training_epochs])
        load_reward = np.zeros([training_epochs])
        X_plot = np.array((range(1, training_epochs + 1)))

        Capacity_DNN = Beamhopping_test(self.Altitude, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.num_timeslot, self.num_angle, self.Earth_radius, self.maximum_transmit_power,
                                     self.carrier_frequency, self.user_antenna_gain, self.bandwidth, self.dB3_angle, self.noise, self.elevation_angle_candidate, self.R_min, self.average_demand,
                                     2000, 17, [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.0001], [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.0001], [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.0001],
                                        [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.0001], 5, self.ckpt)

        iter_num = 0

        for epoch in range(self.num_data_prop):
            train_loss = tf.keras.metrics.Mean()
            train_capacity = tf.keras.metrics.Mean()
            train_reward = tf.keras.metrics.Sum()
            train_power_consumption = tf.keras.metrics.Mean()

            current_timeslot = tf.cast(equal_timeslot + 0.0, tf.float32)
            desired_instance = input_desired[epoch]
            inter_instance = input_inter[epoch]
            demand_instance = input_demand[epoch]
            channel_instance = input_channel[epoch]
            traffic_rate_instance = input_traffic_rate[epoch]
            input_instance = tf.reshape(input_value[epoch], [self.num_group * self.num_user * self.num_active_beam * self.num_active_beam])

            current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0
            channel_state = tf.where(traffic_rate_instance == 0, -channel_instance, channel_instance / current_rate)
            channel_state = tf.reshape(channel_state, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam)

            state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance)
            state_desired = tf.reduce_sum(state_desired, axis=2)
            state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
            state_inter = tf.reduce_sum(state_inter, axis=2)
            current_state = state_desired / state_inter
            current_state = tf.reshape(current_state, state_dim)

            power_matrix = tf.linalg.inv(tf.reshape(channel_state, [self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
            power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 1, 2)
            power_value = tf.reshape(power_value, [self.num_fixed_beam * self.num_user])
            power_value = tf.where(power_value < 0, Pmax_val, power_value)

            timeslot_for_state = tf.tile(tf.reshape(current_timeslot, [self.num_group, 1, 1]), [1, self.num_active_beam, self.num_user])
            timeslot_for_state = tf.reshape(timeslot_for_state, [self.num_fixed_beam * self.num_user])

            traffic_rate = 2.0 ** (demand_instance * self.num_user * self.num_timeslot / timeslot_for_state) - 1.0
            equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user / self.num_timeslot
            desired_SNR = desired_instance
            inter_SNR = tf.reduce_sum(inter_instance * equal_power, -1) + 1.0

            current_reward = 10.0 * np.log10(tf.reduce_sum(timeslot_for_state / self.num_timeslot * traffic_rate * inter_SNR / desired_SNR, -1))

            start_time = time.time()
            done = 0

            for ep in range(max_episodes):
                current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0
                # current_state = tf.concat([input_instance, current_timeslot / (self.num_timeslot / self.num_group)], -1)
                current_rate2 = tf.reshape(tf.reduce_sum(traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1]), axis=2), state_dim)

                state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance)
                state_desired = tf.reduce_sum(state_desired, axis=2)
                state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
                state_inter = tf.reduce_sum(state_inter, axis=2)
                current_state = state_desired / state_inter
                current_state = tf.reshape(tf.math.log(1.0 + current_state) / tf.math.log(2.0), state_dim)
                current_state = tf.concat([current_state, current_rate2], -1)
                # current_state = current_state / current_rate2

                current_action = self.get_action3(current_state, state_dim * 2, action_dim, epsilon, model)

                while True:
                    cal_timeslot = current_timeslot + tf.reduce_sum(tf.expand_dims(tf.cast(current_action, tf.float32), -1) * action_value, 0)
                    if any(cal_timeslot == 0) == True:
                        random_action = np.zeros([action_dim])
                        randint = np.random.randint(action_dim)
                        random_action[randint] = 1
                        current_action = random_action

                    if any(cal_timeslot == 0) != True:
                        break

                # if (ep == (max_episodes - 1)) or (np.sum(current_action == zero_action) == action_dim):
                #    done = 1
                if np.sum(current_action == zero_action) == action_dim:
                    done = 1

                current_timeslot = tf.where(any(cal_timeslot == 0), current_timeslot, cal_timeslot)
                current_action = tf.where(any(cal_timeslot == 0), zero_action, current_action)
                current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0

                channel_state = tf.where(traffic_rate_instance == 0, -channel_instance, channel_instance / current_rate)
                channel_state = tf.reshape(channel_state, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam)
                power_matrix = tf.linalg.inv(tf.reshape(channel_state, [self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
                power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 1, 2)
                power_value = tf.reshape(power_value, [self.num_fixed_beam * self.num_user])
                equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user / self.num_timeslot
                power_value = tf.where(power_value < 0, equal_power, power_value)
                power_value = tf.where(np.sum(power_value) > Pmax_val, power_value / np.sum(power_value) * Pmax_val * self.num_timeslot / timeslot_for_state, power_value)

                timeslot_for_state = tf.tile(tf.reshape(current_timeslot, [self.num_group, 1, 1]), [1, self.num_active_beam, self.num_user])
                timeslot_for_state = tf.reshape(timeslot_for_state, [self.num_fixed_beam * self.num_user])

                inter_SNR = tf.reduce_sum(inter_instance * power_value, -1) + 1.0

                desired_SNR = desired_instance * power_value
                capacity = tf.reduce_sum(self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0), -1)
                capacity_gap = tf.math.abs(self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0) - self.bandwidth * demand_instance)

                state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance)
                state_desired = tf.reduce_sum(state_desired, axis=2)
                state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
                state_inter = tf.reduce_sum(state_inter, axis=2)
                current_rate2 = tf.reshape(tf.reduce_sum(traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1]), axis=2), state_dim)
                next_state = state_desired / state_inter
                next_state = tf.reshape(tf.math.log(1.0 + next_state) / tf.math.log(2.0), state_dim)
                next_state = tf.concat([next_state, current_rate2], -1)
                # next_state = next_state / current_rate2

                previous_reward = current_reward

                traffic_rate = 2.0 ** (demand_instance * self.num_user * self.num_timeslot / timeslot_for_state) - 1.0
                equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user / self.num_timeslot
                desired_SNR = desired_instance
                inter_SNR = tf.reduce_sum(inter_instance * equal_power, -1) + 1.0

                current_reward = 10.0 * np.log10(tf.reduce_sum(timeslot_for_state / self.num_timeslot * traffic_rate * inter_SNR / desired_SNR, -1))

                put_reward = (previous_reward / current_reward) - 1.0
                put_reward = tf.math.sign(put_reward)
                # put_reward = self.maximum_transmit_power - current_reward

                buffer.put(current_state, current_action, put_reward, next_state, done)
                train_capacity(tf.reduce_sum(capacity_gap))
                train_reward(put_reward)
                train_power_consumption(current_reward)

                if buffer.size() >= self.batch_size:
                    self.replay(buffer, state_dim * 2, model, target_model, discount_factor, optimizer, train_loss)

                if done == 1:
                    break

            total_time = time.time() - start_time
            print(current_timeslot)

            epsilon = np.maximum(eps_min, epsilon * eps_decay)

            if epoch % 10 == 0 and epoch > 0 and buffer.size() >= self.batch_size:
                self.target_update(model, target_model)

            total_time = time.time() - start_time
            template = 'DQN-based timeslot training (graph): \n epoch : {}, loss : {:.3f}, gap : {:.3f} [Mbps], cumulative reward : {:.3f}, power consumption : {:.3f} [dBm], time : {:.3f}'
            print(template.format(epoch + 1, train_loss.result(), train_capacity.result(), train_reward.result(), 10.0 * np.log10(train_power_consumption.result()), total_time))

            test_current_timeslot = tf.tile(tf.expand_dims(tf.cast(equal_timeslot, tf.float32), 0), [test_num_data_prop, 1])

            max_episodes = self.num_timeslot - self.num_group
            #model2 = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, 3)
            #model2.load_weights(self.ckpt[3])  # restore checkpoint file

            if epoch % 10 == 0:
                for ep in range(max_episodes):
                    test_current_rate = 2.0 ** (tf_input_traffic_rate * self.num_user * self.num_timeslot / tf.reshape(test_current_timeslot, [test_num_data_prop, self.num_group, 1, 1, 1])) - 1.0
                    test_current_rate2 = tf.reshape(tf.reduce_sum(tf_input_traffic_rate * self.num_user * self.num_timeslot / tf.reshape(test_current_timeslot, [test_num_data_prop, self.num_group, 1, 1, 1]), axis=3), [test_num_data_prop, state_dim])

                    test_state_desired = tf.where(tf_input_traffic_rate == 0, 0.0, tf_input_channel)
                    test_state_desired = tf.reduce_sum(test_state_desired, axis=3)
                    test_state_inter = tf.where(tf_input_traffic_rate == 0, tf_input_channel, 0.0)
                    test_state_inter = tf.reduce_sum(test_state_inter, axis=3)
                    test_current_state = test_state_desired / test_state_inter
                    test_current_state = tf.reshape(tf.math.log(1.0 + test_current_state) / tf.math.log(2.0),
                                               [test_num_data_prop, state_dim])
                    test_current_state = tf.concat([test_current_state, test_current_rate2], -1)

                    test_current_action = tf.stop_gradient(model.DQN_action(test_current_state))
                    test_current_action = tf.cast(tf.reduce_max(test_current_action, axis=-1, keepdims=True) == test_current_action, tf.float32)

                    test_cal_timeslot = test_current_timeslot + tf.reduce_sum(tf.expand_dims(tf.cast(test_current_action, tf.float32), -1) * action_value, 1)
                    test_cal_timeslot2 = tf.expand_dims(tf.reduce_sum(tf.cast(test_cal_timeslot == 0, tf.float32), -1), -1)
                    test_current_timeslot = tf.where(test_cal_timeslot2 == 0, test_cal_timeslot, test_current_timeslot)

                desired_signal = test_group_channel_gain * np.expand_dims(test_snapshot_index, (1, 4)) * np.expand_dims(test_group_snapshot_index, (2, 4))
                inter_signal = test_group_channel_gain * np.expand_dims(test_group_snapshot_index, (2, 4)) * np.expand_dims(test_group_snapshot_index, (3, 4)) - desired_signal

                input_desired_signal = np.reshape(np.sum(desired_signal, (1, 3)), [-1, self.num_fixed_beam * self.num_user])
                input_inter_signal = np.reshape(np.swapaxes(np.sum(inter_signal, 1), 2, 3), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam])
                input_inter_signal = np.reshape(np.tile(np.expand_dims(input_inter_signal, -1), [1, 1, 1, self.num_user]), [-1, self.num_fixed_beam * self.num_user, self.num_fixed_beam * self.num_user])
                eye_element = np.expand_dims(np.tile(np.eye(self.num_user), [self.num_fixed_beam, self.num_fixed_beam]), 0)
                input_inter_signal = input_inter_signal * eye_element
                test_timeslot_for_state = np.tile(tf.reshape(test_current_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
                test_timeslot_for_state = np.reshape(test_timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])
                equal_power = Pmax_val * np.ones([test_num_data_prop, self.num_fixed_beam * self.num_user]) / self.total_user / self.num_timeslot
                desired_SNR = input_desired_signal
                inter_SNR = np.sum(input_inter_signal * np.expand_dims(equal_power, 1), -1) + 1.0

                test_current_timeslot2 = np.tile(np.reshape(test_current_timeslot, [test_num_data_prop, self.num_group, 1, 1]), [1,1,self.num_active_beam,self.num_user])
                test_current_timeslot2 = np.reshape(test_current_timeslot2, [-1,self.num_fixed_beam*self.num_user])

                test_current_rate = 2.0 ** (np.reshape(test_input_traffic_demand, [-1,self.num_fixed_beam*self.num_user]) / self.bandwidth * self.num_user / test_current_timeslot2 * self.num_timeslot) - 1.0

                test_reward = np.sum(test_timeslot_for_state / self.num_timeslot * np.reshape(test_current_rate, [-1,self.num_fixed_beam*self.num_user]) * inter_SNR / desired_SNR, -1)

                load_reward[iter_num] = np.array(10.0 * np.log10(np.mean(test_reward)))

                DNN_input1, DNN_input2, DNN_input3 = self.Normalized_input(test_group_channel_gain, test_snapshot_index, test_group_snapshot_index, tf_input_traffic_demand, test_current_timeslot)
                minimum_power = self.matrix_inversion_based_power_allocation(test_input_traffic_rate, test_input_channel, test_current_timeslot)
                minimum_power_alloc = np.expand_dims(minimum_power, (1, 2))
                minimum_capacity = self.data_rate_proposed(test_group_channel_gain, minimum_power_alloc, test_snapshot_index, test_group_snapshot_index, test_current_timeslot)

                capacity_min, capacity_max = self.capacity_min_max(test_input_traffic_rate, test_input_channel, test_current_timeslot, minimum_capacity, test_num_data_prop)

                DNN_capacity, _ = Capacity_DNN.Unsupervised_Capacity_Estimation_test(test_current_timeslot, DNN_input3, capacity_min, capacity_max)
                DNN_capacity = np.reshape(DNN_capacity, [-1, self.total_user]) * self.bandwidth
                input_traffic_rate2 = np.swapaxes(tf.reshape(DNN_capacity / self.bandwidth, [-1, self.num_group, self.num_active_beam, self.num_user]), 2, 3)
                input_traffic_rate2 = np.expand_dims(input_traffic_rate2, 3) * np.eye(self.num_active_beam)

                DNN_power = self.matrix_inversion_based_power_allocation(input_traffic_rate2, test_input_channel, test_current_timeslot)
                DNN_power = np.reshape(DNN_power, [-1, self.num_fixed_beam, self.num_user])
                power_alloc = np.expand_dims(DNN_power, (1, 2))

                prop_data_rate = self.data_rate_proposed(test_group_channel_gain, power_alloc, test_snapshot_index, test_group_snapshot_index, test_current_timeslot)
                outage_rate = self.outage_rate(prop_data_rate, tf_input_traffic_demand, test_num_data_prop)
                energy_efficiency = self.energy_efficiency(prop_data_rate, DNN_power, test_current_timeslot, test_group_snapshot_index)
                EE = np.mean(energy_efficiency)

                load_outage[iter_num] = outage_rate
                load_EE[iter_num] = EE

                plt.figure(1)
                plt.plot(X_plot[:iter_num], load_EE[:iter_num], c='b')
                plt.xticks(np.array(range(11)) * int(training_epochs/100))
                plt.xlabel('Epoch')
                plt.ylabel('Energy efficiency [Mbps/J]')
                plt.legend(['Proposed DQN'])
                plt.grid(visible=True)
                plt.pause(0.001)

                plt.figure(2)
                plt.plot(X_plot[:iter_num], load_outage[:iter_num]/100.0, c='b')
                plt.xticks(np.array(range(11)) * int(training_epochs/100))
                plt.xlabel('Epoch')
                plt.ylabel('outage rate')
                plt.yscale('log')
                plt.legend(['Proposed DQN'])
                plt.grid(visible=True)
                plt.pause(0.001)

                plt.figure(3)
                plt.plot(X_plot[:iter_num], load_reward[:iter_num], c='b')
                plt.xticks(np.array(range(11)) * int(training_epochs/100))
                plt.xlabel('Epoch')
                plt.ylabel('Reward [dBm]')
                plt.legend(['Proposed DQN'])
                plt.grid(visible=True)
                plt.pause(0.001)

                iter_num = iter_num + 1

        np.savetxt(str(self.ckpt[self.DL_scheme_val]) + '_EE.csv', load_EE, delimiter=",")
        np.savetxt(str(self.ckpt[self.DL_scheme_val]) + '_outage.csv', load_outage, delimiter=",")
        np.savetxt(str(self.ckpt[self.DL_scheme_val]) + '_reward.csv', load_reward, delimiter=",")

        return 0

    def replay2(self, buffer, state_dim, model, target_model, discount_factor, optimizer, train_loss):
        states, actions, rewards, next_states, done = buffer.sample(self.batch_size)
        states = np.reshape(states, [-1, state_dim])
        next_states = np.reshape(next_states, [-1, state_dim])
        next_q_values = np.max(target_model(next_states), axis=-1)

        targets = rewards + (1-done) * next_q_values * discount_factor

        self.Reward_Optimized_DQN_Timeslot_Allocation_train_step(states, actions, targets, model, optimizer, train_loss)

    def get_action2(self, state, state_dim, action_dim, epsilon, model):
        state = np.reshape(state, [1, state_dim])

        if np.random.random() < epsilon:
            random_action = np.zeros([action_dim])
            randint = np.random.randint(action_dim)
            random_action[randint] = 1

            return random_action

        else:
            q_value = model(state)[0]
            #print(q_value)
            max_q = np.max(q_value, keepdims=True)
            action_index = q_value == max_q

            return action_index

    def get_action3(self, state, state_dim, action_dim, epsilon, model):
        state = np.reshape(state, [1, state_dim])

        if np.random.random() < epsilon:
            random_action = np.zeros([action_dim])
            randint = np.random.randint(action_dim)
            random_action[randint] = 1

            return random_action

        else:
            q_value = model.DQN_action(state)[0]
            #print(q_value)
            max_q = np.max(q_value, keepdims=True)
            action_index = q_value == max_q

            return action_index

    def Single_DQN_Timeslot_Allocation_train2(self, input_desired, input_inter, input_demand, epsilon, eps_decay, eps_min, discount_factor, input_channel, input_traffic_rate, input_value):
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        batch_prob = True
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        lam = self.lambda_val[self.DL_scheme_val]
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        input_value = tf.cast(input_value, tf.float32)

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        target_model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)

        len_capacity = 1000
        buffer = ReplayBuffer(len_capacity)

        max_episodes = self.num_timeslot

        if self.reuse == 1:
            model.load_weights(self.ckpt[self.DL_scheme_val])  # restore checkpoint file
            target_model.load_weights(self.ckpt[self.DL_scheme_val])

        if self.reuse == 0:
            self.target_update(model, target_model)

        # state_dim = int(self.num_group*self.num_user*self.num_active_beam*self.num_active_beam + self.num_group)
        state_dim = int(self.num_group * self.num_user * self.num_active_beam)
        action_dim = int(math.perm(self.num_group, 2) + 1)

        action_value_0 = np.zeros(self.num_group)
        action_value_sample = action_value_0 + 0
        action_value_sample[0] = 1
        action_value_sample[-1] = -1

        action_value = np.array(list(itertools.permutations(action_value_sample, self.num_group)))
        action_value = np.unique(action_value, axis=0)
        action_value = np.concatenate([np.expand_dims(action_value_0, 0), action_value], axis=0)
        action_value = tf.cast(action_value, tf.float32)

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot

        zero_action = np.zeros(action_dim)
        zero_action[0] = 1

        eps_count = 0

        for epoch in range(self.num_data_prop):
            train_loss = tf.keras.metrics.Mean()
            train_capacity = tf.keras.metrics.Mean()
            train_reward = tf.keras.metrics.Sum()
            train_power_consumption = tf.keras.metrics.Mean()

            current_timeslot = tf.cast(equal_timeslot + 0.0, tf.float32)
            desired_instance = input_desired[epoch]
            inter_instance = input_inter[epoch]
            demand_instance = input_demand[epoch]
            channel_instance = input_channel[epoch]
            traffic_rate_instance = input_traffic_rate[epoch]
            input_instance = tf.reshape(input_value[epoch], [self.num_group * self.num_user * self.num_active_beam * self.num_active_beam])

            current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0
            channel_state = tf.where(traffic_rate_instance == 0, -channel_instance, channel_instance / current_rate)
            channel_state = tf.reshape(channel_state, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam)

            state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance)
            state_desired = tf.reduce_sum(state_desired, axis=2)
            state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
            state_inter = tf.reduce_sum(state_inter, axis=2)
            current_state = state_desired / state_inter
            current_state = tf.reshape(current_state, state_dim)

            power_matrix = tf.linalg.inv(tf.reshape(channel_state, [self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
            power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 1, 2)
            power_value = tf.reshape(power_value, [self.num_fixed_beam * self.num_user])
            power_value = tf.where(power_value < 0, Pmax_val, power_value)

            timeslot_for_state = tf.tile(tf.reshape(current_timeslot, [self.num_group, 1, 1]), [1, self.num_active_beam, self.num_user])
            timeslot_for_state = tf.reshape(timeslot_for_state, [self.num_fixed_beam * self.num_user])

            traffic_rate = 2.0 ** (demand_instance * self.num_user * self.num_timeslot / timeslot_for_state) - 1.0
            equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user / self.num_timeslot
            desired_SNR = desired_instance
            inter_SNR = tf.reduce_sum(inter_instance * equal_power, -1) + 1.0

            current_reward = 10.0 * np.log10(tf.reduce_sum(timeslot_for_state / self.num_timeslot * traffic_rate * inter_SNR / desired_SNR, -1))

            start_time = time.time()
            done = 0
            for ep in range(max_episodes):
                current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0
                # current_state = tf.concat([input_instance, current_timeslot / (self.num_timeslot / self.num_group)], -1)
                current_rate2 = tf.reshape(tf.reduce_sum(traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1]), axis=2), state_dim)

                state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance)
                state_desired = tf.reduce_sum(state_desired, axis=2)
                state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
                state_inter = tf.reduce_sum(state_inter, axis=2)
                current_state = state_desired / state_inter
                current_state = tf.reshape(tf.math.log(1.0 + current_state) / tf.math.log(2.0), state_dim)
                current_state = tf.concat([current_state, current_rate2], -1)
                #current_state = current_state / current_rate2

                current_action = self.get_action3(current_state, state_dim*2, action_dim, epsilon, model)

                while True:
                    cal_timeslot = current_timeslot + tf.reduce_sum(tf.expand_dims(tf.cast(current_action, tf.float32), -1) * action_value, 0)
                    if any(cal_timeslot == 0) == True:
                        random_action = np.zeros([action_dim])
                        randint = np.random.randint(action_dim)
                        random_action[randint] = 1
                        current_action = random_action

                    if any(cal_timeslot == 0) != True:
                        break

                #if (ep == (max_episodes - 1)) or (np.sum(current_action == zero_action) == action_dim):
                #    done = 1
                if np.sum(current_action == zero_action) == action_dim:
                    done = 1

                current_timeslot = tf.where(any(cal_timeslot == 0), current_timeslot, cal_timeslot)
                current_action = tf.where(any(cal_timeslot == 0), zero_action, current_action)
                current_rate = 2.0 ** (traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1])) - 1.0

                channel_state = tf.where(traffic_rate_instance == 0, -channel_instance, channel_instance / current_rate)
                channel_state = tf.reshape(channel_state, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam)
                power_matrix = tf.linalg.inv(tf.reshape(channel_state, [self.num_group, self.num_user, self.num_active_beam, self.num_active_beam]))
                power_value = tf.experimental.numpy.swapaxes(tf.reduce_sum(power_matrix, -1), 1, 2)
                power_value = tf.reshape(power_value, [self.num_fixed_beam * self.num_user])
                equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user / self.num_timeslot
                power_value = tf.where(power_value < 0, equal_power, power_value)
                power_value = tf.where(np.sum(power_value) > Pmax_val, power_value / np.sum(power_value) * Pmax_val, power_value)

                timeslot_for_state = tf.tile(tf.reshape(current_timeslot, [self.num_group, 1, 1]), [1, self.num_active_beam, self.num_user])
                timeslot_for_state = tf.reshape(timeslot_for_state, [self.num_fixed_beam * self.num_user])

                inter_SNR = tf.reduce_sum(inter_instance * power_value, -1) + 1.0

                desired_SNR = desired_instance * power_value
                capacity = tf.reduce_sum(self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0), -1)
                capacity_gap = tf.math.abs(self.bandwidth / self.num_user * timeslot_for_state / self.num_timeslot * tf.math.log(1.0 + desired_SNR / inter_SNR) / tf.math.log(2.0) - self.bandwidth * demand_instance)

                state_desired = tf.where(traffic_rate_instance == 0, 0.0, channel_instance)
                state_desired = tf.reduce_sum(state_desired, axis=2)
                state_inter = tf.where(traffic_rate_instance == 0, channel_instance, 0.0)
                state_inter = tf.reduce_sum(state_inter, axis=2)
                current_rate2 = tf.reshape(tf.reduce_sum(traffic_rate_instance * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_group, 1, 1, 1]), axis=2), state_dim)
                next_state = state_desired / state_inter
                next_state = tf.reshape(tf.math.log(1.0 + next_state) / tf.math.log(2.0), state_dim)
                next_state = tf.concat([next_state, current_rate2], -1)
                #next_state = next_state / current_rate2

                previous_reward = current_reward

                traffic_rate = 2.0 ** (demand_instance * self.num_user * self.num_timeslot / timeslot_for_state) - 1.0
                equal_power = Pmax_val * tf.ones(self.num_fixed_beam * self.num_user) / self.total_user / self.num_timeslot
                desired_SNR = desired_instance
                inter_SNR = tf.reduce_sum(inter_instance * equal_power, -1) + 1.0

                current_reward = 10.0 * np.log10(tf.reduce_sum(timeslot_for_state / self.num_timeslot * traffic_rate * inter_SNR / desired_SNR, -1))

                put_reward = (previous_reward / current_reward) - 1.0
                put_reward = tf.math.sign(put_reward)
                #put_reward = self.maximum_transmit_power - current_reward

                buffer.put(current_state, current_action, put_reward, next_state, done)
                train_capacity(tf.reduce_sum(capacity_gap))
                train_reward(put_reward)
                train_power_consumption(current_reward)

                if buffer.size() >= self.batch_size:
                    self.replay(buffer, state_dim*2, model, target_model, discount_factor, optimizer, train_loss)

                if done == 1:
                    break

            total_time = time.time() - start_time
            print(current_timeslot)

            epsilon = np.maximum(eps_min, epsilon * eps_decay)

            if epoch % 10 == 0 and epoch > 0 and buffer.size() >= self.batch_size:
                self.target_update(model, target_model)

            template = 'DQN-based timeslot training (only timeslot): \n epoch : {}, loss : {:.3f}, gap : {:.3f} [Mbps], cumulative reward : {:.3f}, power consumption : {:.3f} [dBm], time : {:.3f}'
            print(template.format(epoch + 1, train_loss.result(), train_capacity.result(), train_reward.result(), 10.0 * np.log10(train_power_consumption.result()), total_time))

        model.save_weights(self.ckpt[self.DL_scheme_val])

        return 0

class Beamhopping_test:
    def __init__(self, Altitude, num_fixed_beam, num_active_beam, num_group, num_user, num_timeslot, num_angle, Earth_radius, maximum_transmit_power, carrier_frequency, user_antenna_gain, bandwidth,
                 dB3_angle, noise, elevation_angle_candidate, R_min, average_demand, num_instances, seed, learning_rate, training_epochs, batch_size, lambda_val, DL_scheme_val, ckpt):
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

        self.learning_rate = learning_rate[DL_scheme_val]
        self.training_epochs = training_epochs[DL_scheme_val]
        self.batch_size = batch_size[DL_scheme_val]
        self.lambda_val = lambda_val
        self.DL_scheme_val = DL_scheme_val

        self.ckpt = ckpt
        # set parameter

    #@tf.function
    def Unsupervised_Timeslot_Allocation_test_step(self, model, batch_prob, DNN_input):
        DNN_timeslot = model(DNN_input, batch_prob)

        return DNN_timeslot

    def Unsupervised_Timeslot_Allocation_test(self, input_desired, input_inter, input_demand):
        batch_prob = False
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)

        desired_concat = tf.expand_dims(input_desired, -1)
        inter_concat = input_inter
        demand_concat = tf.expand_dims(input_demand, -1)
        DNN_input = tf.concat([desired_concat,inter_concat,demand_concat], -1)
        DNN_input = tf.reshape(DNN_input, [-1,self.num_fixed_beam*self.num_user*(self.num_fixed_beam*self.num_user+2)])
        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        model.load_weights(self.ckpt[self.DL_scheme_val])
        start_time = time.time()
        DNN_timeslot = self.Unsupervised_Timeslot_Allocation_test_step(model, batch_prob, DNN_input)
        end_time = time.time() - start_time

        return DNN_timeslot, end_time

    def Unsupervised_Power_Allocation_test(self, input_timeslot, DNN_input, minimum_power):
        batch_prob = False
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        timeslot_for_state = tf.tile(tf.reshape(input_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_for_state = tf.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        model.load_weights(self.ckpt[self.DL_scheme_val])
        start_time = time.time()
        DNN_power = model(DNN_input, batch_prob, timeslot_for_state, minimum_power)
        end_time = time.time() - start_time

        return DNN_power, end_time

    def Unsupervised_Capacity_Estimation_test(self, input_timeslot, DNN_input, capacity_min, capacity_max):
        batch_prob = False
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        timeslot_for_state = tf.tile(tf.reshape(input_timeslot, [-1, self.num_group, 1, 1]), [1, 1, self.num_active_beam, self.num_user])
        timeslot_for_state = tf.reshape(timeslot_for_state, [-1, self.num_fixed_beam * self.num_user])

        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group, self.num_user, self.DL_scheme_val)
        model.load_weights(self.ckpt[self.DL_scheme_val])
        start_time = time.time()
        DNN_capacity = model(DNN_input, batch_prob, timeslot_for_state, capacity_min, capacity_max)
        end_time = time.time() - start_time

        return DNN_capacity, end_time

    def Single_DQN_Timeslot_Allocation_test_step(self, model, DNN_input):
        DNN_timeslot = model(DNN_input)

        return DNN_timeslot

    def Single_DQN_Timeslot_Allocation_test(self, channel_gain, input_desired, input_inter, input_demand,
                                                      snapshot_index, snapshot_group_index, input_channel,
                                                      input_traffic_rate, input_value):
        batch_prob = False
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group,
                          self.num_user, self.DL_scheme_val)
        model.load_weights(self.ckpt[self.DL_scheme_val])
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        snapshot_index = tf.cast(snapshot_index, tf.float32)
        snapshot_group_index = tf.cast(snapshot_group_index, tf.float32)
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        input_value = tf.cast(input_value, tf.float32)
        input_value = tf.reshape(input_value,
                                 [-1, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam])

        state_dim = int(self.num_group * self.num_user * self.num_active_beam)
        action_dim = int(math.perm(self.num_group, 2) + 1)

        action_value_0 = np.zeros(self.num_group)
        action_value_sample = action_value_0 + 0
        action_value_sample[0] = 1
        action_value_sample[-1] = -1

        action_value = np.array(list(itertools.permutations(action_value_sample, self.num_group)))
        action_value = np.unique(action_value, axis=0)
        action_value = np.concatenate([np.expand_dims(action_value_0, 0), action_value], axis=0)
        action_value = tf.cast(action_value, tf.float32)
        action_value = tf.expand_dims(action_value, 0)

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot
        equal_timeslot = np.tile(np.expand_dims(equal_timeslot, 0), [self.num_data_prop, 1])

        zero_action = np.zeros(action_dim)
        zero_action[0] = 1
        zero_action = np.expand_dims(zero_action, 0)

        current_timeslot = tf.cast(equal_timeslot, tf.float32)

        max_episodes = self.num_timeslot - self.num_group
        start_time = time.time()
        for ep in range(max_episodes):
            current_rate = 2.0 ** (input_traffic_rate * self.num_user / tf.reshape(current_timeslot,
                                                                                   [self.num_data_prop, self.num_group,
                                                                                    1, 1, 1])) - 1.0
            # current_state = tf.concat([input_instance, current_timeslot / (self.num_timeslot / self.num_group)], -1)

            state_desired = tf.where(input_traffic_rate == 0, 0.0, input_channel / current_rate)
            state_desired = tf.reduce_sum(state_desired, axis=3)
            state_inter = tf.where(input_traffic_rate == 0, input_channel, 0.0)
            state_inter = tf.reduce_sum(state_inter, axis=3)
            current_state = state_desired / state_inter
            current_state = tf.reshape(tf.math.log(1.0 + current_state) / tf.math.log(10.0),
                                       [self.num_data_prop, state_dim])
            current_state = tf.concat([current_state, current_timeslot / (self.num_timeslot / self.num_group)], -1)

            current_action = self.Single_DQN_Timeslot_Allocation_test_step(model, current_state)
            current_action = tf.cast(tf.reduce_max(current_action, axis=-1, keepdims=True) == current_action,
                                     tf.float32)

            cal_timeslot = current_timeslot + tf.reduce_sum(
                tf.expand_dims(tf.cast(current_action, tf.float32), -1) * action_value, 1)
            cal_timeslot2 = tf.expand_dims(tf.reduce_sum(tf.cast(cal_timeslot == 0, tf.float32), -1), -1)
            current_timeslot = tf.where(cal_timeslot2 == 0, cal_timeslot, current_timeslot)

        end_time = time.time() - start_time

        return current_timeslot, end_time

    def Single_DQN_Timeslot_Allocation_test2_step(self, model, DNN_input):
        DNN_timeslot = model.DQN_action(DNN_input)

        return DNN_timeslot

    def Single_DQN_Timeslot_Allocation_test2(self, channel_gain, input_desired, input_inter, input_demand,
                                                      snapshot_index, snapshot_group_index, input_channel,
                                                      input_traffic_rate, input_value):
        batch_prob = False
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group,
                          self.num_user, self.DL_scheme_val)
        model.load_weights(self.ckpt[self.DL_scheme_val])
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        snapshot_index = tf.cast(snapshot_index, tf.float32)
        snapshot_group_index = tf.cast(snapshot_group_index, tf.float32)
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        input_value = tf.cast(input_value, tf.float32)
        input_value = tf.reshape(input_value,
                                 [-1, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam])

        state_dim = int(self.num_group * self.num_user * self.num_active_beam)
        action_dim = int(math.perm(self.num_group, 2) + 1)

        action_value_0 = np.zeros(self.num_group)
        action_value_sample = action_value_0 + 0
        action_value_sample[0] = 1
        action_value_sample[-1] = -1

        action_value = np.array(list(itertools.permutations(action_value_sample, self.num_group)))
        action_value = np.unique(action_value, axis=0)
        action_value = np.concatenate([np.expand_dims(action_value_0, 0), action_value], axis=0)
        action_value = tf.cast(action_value, tf.float32)
        action_value = tf.expand_dims(action_value, 0)

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot
        equal_timeslot = np.tile(np.expand_dims(equal_timeslot, 0), [self.num_data_prop, 1])

        zero_action = np.zeros(action_dim)
        zero_action[0] = 1
        zero_action = np.expand_dims(zero_action, 0)

        current_timeslot = tf.cast(equal_timeslot, tf.float32)

        max_episodes = self.num_timeslot - self.num_group
        start_time = time.time()
        for ep in range(max_episodes):
            current_rate = 2.0 ** (input_traffic_rate * self.num_user / tf.reshape(current_timeslot,
                                                                                   [self.num_data_prop, self.num_group,
                                                                                    1, 1, 1])) - 1.0
            current_rate2 = tf.reshape(tf.reduce_sum(input_traffic_rate * self.num_user * self.num_timeslot / tf.reshape(current_timeslot, [self.num_data_prop, self.num_group, 1, 1, 1]), axis=3), [-1, state_dim])
            # current_state = tf.concat([input_instance, current_timeslot / (self.num_timeslot / self.num_group)], -1)

            state_desired = tf.where(input_traffic_rate == 0, 0.0, input_channel)
            state_desired = tf.reduce_sum(state_desired, axis=3)
            state_inter = tf.where(input_traffic_rate == 0, input_channel, 0.0)
            state_inter = tf.reduce_sum(state_inter, axis=3)
            current_state = state_desired / state_inter
            current_state = tf.reshape(tf.math.log(1.0 + current_state) / tf.math.log(2.0),
                                       [self.num_data_prop, state_dim])
            current_state = tf.concat([current_state, current_rate2], -1)
            #current_state = current_state / current_rate2

            current_action = self.Single_DQN_Timeslot_Allocation_test2_step(model, current_state)
            current_action = tf.cast(tf.reduce_max(current_action, axis=-1, keepdims=True) == current_action,
                                     tf.float32)

            cal_timeslot = current_timeslot + tf.reduce_sum(
                tf.expand_dims(tf.cast(current_action, tf.float32), -1) * action_value, 1)
            cal_timeslot2 = tf.expand_dims(tf.reduce_sum(tf.cast(cal_timeslot == 0, tf.float32), -1), -1)
            current_timeslot = tf.where(cal_timeslot2 == 0, cal_timeslot, current_timeslot)

        end_time = time.time() - start_time

        return current_timeslot, end_time

    def Reward_Optimized_DQN_Timeslot_Allocation_test_step(self, model, DNN_input):
        DNN_timeslot = model(DNN_input)

        return DNN_timeslot

    def Reward_Optimized_DQN_Timeslot_Allocation_test(self, channel_gain, input_desired, input_inter, input_demand,
                                                      snapshot_index, snapshot_group_index, input_channel,
                                                      input_traffic_rate, input_value):
        batch_prob = False
        Pmax_val = 10.0 ** (self.maximum_transmit_power / 10.0)
        model = DNN_model(Pmax_val, self.num_timeslot, self.num_fixed_beam, self.num_active_beam, self.num_group,
                          self.num_user, self.DL_scheme_val)
        model.load_weights(self.ckpt[self.DL_scheme_val])
        input_desired = tf.cast(input_desired, tf.float32)
        input_inter = tf.cast(input_inter, tf.float32)
        input_demand = tf.cast(input_demand, tf.float32)
        snapshot_index = tf.cast(snapshot_index, tf.float32)
        snapshot_group_index = tf.cast(snapshot_group_index, tf.float32)
        input_channel = tf.cast(input_channel, tf.float32)
        input_traffic_rate = tf.cast(input_traffic_rate, tf.float32)
        input_value = tf.cast(input_value, tf.float32)
        input_value = tf.reshape(input_value,
                                 [-1, self.num_group * self.num_user * self.num_active_beam * self.num_active_beam])

        state_dim = int(self.num_group * self.num_user * self.num_active_beam)
        action_dim = int(math.perm(self.num_group, 2) + 1)

        action_value_0 = np.zeros(self.num_group)
        action_value_sample = action_value_0 + 0
        action_value_sample[0] = 1
        action_value_sample[-1] = -1

        action_value = np.array(list(itertools.permutations(action_value_sample, self.num_group)))
        action_value = np.unique(action_value, axis=0)
        action_value = np.concatenate([np.expand_dims(action_value_0, 0), action_value], axis=0)
        action_value = tf.cast(action_value, tf.float32)
        action_value = tf.expand_dims(action_value, 0)

        equal_timeslot = int(self.num_timeslot / self.num_group) * np.ones(self.num_group)
        extra_timeslot = self.num_timeslot - np.sum(equal_timeslot)
        equal_timeslot[0] = equal_timeslot[0] + extra_timeslot
        equal_timeslot = np.tile(np.expand_dims(equal_timeslot, 0), [self.num_data_prop, 1])

        zero_action = np.zeros(action_dim)
        zero_action[0] = 1
        zero_action = np.expand_dims(zero_action, 0)

        current_timeslot = tf.cast(equal_timeslot, tf.float32)

        max_episodes = self.num_timeslot - self.num_group
        start_time = time.time()
        for ep in range(max_episodes):
            current_rate = 2.0 ** (input_traffic_rate * self.num_user / tf.reshape(current_timeslot,
                                                                                   [self.num_data_prop, self.num_group,
                                                                                    1, 1, 1])) - 1.0
            # current_state = tf.concat([input_instance, current_timeslot / (self.num_timeslot / self.num_group)], -1)

            state_desired = tf.where(input_traffic_rate == 0, 0.0, input_channel / current_rate)
            state_desired = tf.reduce_sum(state_desired, axis=3)
            state_inter = tf.where(input_traffic_rate == 0, input_channel, 0.0)
            state_inter = tf.reduce_sum(state_inter, axis=3)
            current_state = state_desired / state_inter
            current_state = tf.reshape(tf.math.log(1.0 + current_state) / tf.math.log(10.0),
                                       [self.num_data_prop, state_dim])
            current_state = tf.concat([current_state, current_timeslot / (self.num_timeslot / self.num_group)], -1)

            current_action = self.Reward_Optimized_DQN_Timeslot_Allocation_test_step(model, current_state)
            current_action = tf.cast(tf.reduce_max(current_action, axis=-1, keepdims=True) == current_action,
                                     tf.float32)

            cal_timeslot = current_timeslot + tf.reduce_sum(
                tf.expand_dims(tf.cast(current_action, tf.float32), -1) * action_value, 1)
            cal_timeslot2 = tf.expand_dims(tf.reduce_sum(tf.cast(cal_timeslot == 0, tf.float32), -1), -1)
            current_timeslot = tf.where(cal_timeslot2 == 0, cal_timeslot, current_timeslot)

        end_time = time.time() - start_time

        return current_timeslot, end_time
