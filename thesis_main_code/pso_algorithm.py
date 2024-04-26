import numpy as np
from forward_kinematics import piecewise_cc


class ParticleSwarmOptimization:
    """
    Class includes method optimize for PSO which returns configuration space variables (theta, phi)

    Sources:
    [1] https://gist.github.com/ljvmiranda921/7d8c48da0aa7565f0b3c01d7c951c5e9
    [2] https://pyswarms.readthedocs.io/en/development/examples/inverse_kinematics.html
    """
    def __init__(self):
        self.current_pos = None
        self.p_best_pos = None
        self.g_best_pos = None
        self.velocity = None

    def optimize(self,
                 num_seg: int,
                 seg_len: np.ndarray[float],
                 num_of_el: np.ndarray[int],
                 di: float,
                 angle_limits: np.ndarray[int],
                 target_pos: np.ndarray[float]) -> np.ndarray[float]:
        """
        Function search for possible solution of Inverse Kinematics.

        :param num_seg: Number of segments
        :param seg_len: Lengths of the segments  [m]
        :param num_of_el: Number of elements per segment
            if n=1 all segments with equal number of points
        :param di: Arc end connection distance from origin of local coordinate system [m]
        :param angle_limits: array with theta max and phi max (starting from zero) in degrees.
        :param target_pos: Coordinates of the target point (Ex, Ey, Ez)

        :return: final_params: Configuration space angles
        """
        num_seg = num_seg
        seg_len = seg_len
        num_of_el = num_of_el
        di = di
        target_pos = target_pos
        #                                            SETUP PARAMETERS                                            #
        swarm_size = 15
        max_iter = 45
        params = np.zeros((num_seg * 2, 1))  # start with all angles set to 0
        min_error = 0.0001  # in meters
        influence = {'c1': 1.2, 'c2': 1.2}  # c1 - personal, c2 - social
        bounds = {'theta_min': 0,
                  'theta_max': angle_limits[0],
                  'phi_min': 0,
                  'phi_max': angle_limits[1]}  # boundary conditions for every section of CR
        v_initial = np.zeros((1, num_seg * 2))  # initial velocity starts wi

        #                                             HELPER FUNCTIONS                                            #
        def boundary_condition(element, algorithm):
            """
            Function applies lower and upper bound limits.

            :param element: n dimensional particle
            :param algorithm: 'x' or 'v' selects whether you want to update velocity or position
            """
            if algorithm == 'x':
                self.current_pos[element, :num_seg] = np.clip(self.current_pos[element, :num_seg], bounds['theta_min'],
                                                              bounds['theta_max'])  # second half are phis
                self.current_pos[element, num_seg:] = np.clip(self.current_pos[element, num_seg:], bounds['phi_min'],
                                                              bounds['phi_max'])  # second half are phis
            if algorithm == 'v':
                theta_v_max = bounds['theta_max'] - bounds['theta_min']
                theta_v_min = - theta_v_max
                phi_v_max = bounds['phi_max'] - bounds['phi_min']
                phi_v_min = - phi_v_max
                self.velocity[element, :num_seg] = np.clip(self.velocity[element, :num_seg], theta_v_min,
                                                           theta_v_max)  # second half are phis
                self.velocity[element, num_seg:] = np.clip(self.velocity[element, num_seg:], phi_v_min,
                                                           phi_v_max)  # second half are phis

        def end_tip_position(parameters):
            """
            Calculates forward kinematics and returns last's segment endpoint position.

            :param parameters: [np.array] Input angle configuration space variables (theta, phi)

            :return ep_pos: [np.array] Endpoint position [Ex, Ey, Ez]
            """
            end_pos = piecewise_cc(num_seg=num_seg,
                                   theta=parameters[:num_seg],  # the first half of  the list are theta's
                                   phi=np.deg2rad(parameters[num_seg:]),  # the second half of the list are phi's
                                   seg_len=seg_len,
                                   di=di,
                                   num_of_el=num_of_el,
                                   optimizer=True)
            return end_pos

        def objective_function(X, target):
            """
            Calculates the difference between two points in space and returns error vector norm.

            :param X: [np.array] Population of swarm
            :param target: [np.array] Coordinates of the target point (Ex, Ey, Ez)

            :return error: Calculated error
            """
            test_pos = end_tip_position(X)
            x_dist = target[0] - test_pos[0]
            y_dist = target[1] - test_pos[1]
            z_dist = target[2] - test_pos[2]
            error = np.linalg.norm([x_dist, y_dist, z_dist])
            return error

        def inertia_weight_update(iteration, max_iteration):
            """Changes inertia weight during the iteration from w_max to w_min, returns computed w.

            :param iteration: Actual iteration start from 1
            :param max_iteration: Maximal number of iterations

            :return w: Inertia weight constant
            """
            w_min = 0.4
            w_max = 0.9
            w = w_max - (w_max-w_min)/max_iteration*iteration
            return w

        def initialization():
            """
            Initialization of PSO, generating random position for x_i, initialize velocity with zeros, definition of
            current position (current_pos), personal best position(p_best_pos), global best position (g_best_pos).
            """
            # Random position 'angle value' for particles
            current_pos_theta = np.random.uniform(bounds['theta_min'], bounds['theta_max'], [swarm_size, num_seg])
            current_pos_phi = np.random.uniform(bounds['phi_min'], bounds['phi_max'], [swarm_size, num_seg])
            self.current_pos = np.concatenate((current_pos_theta, current_pos_phi), axis=1)
            # Current best particle position
            self.p_best_pos = self.current_pos.copy()
            # Current best global position
            J = []
            for j in range(swarm_size):
                temp = objective_function(self.current_pos[j], target_pos)  # temp = error
                J.append(temp)
            min_error_id = np.argmin(J)  # min error index in J list
            self.g_best_pos = self.current_pos[min_error_id, :]  # saving min_error population into g_best_pos
            # Initialize Velocity
            self.velocity = v_initial * np.ones([swarm_size, params.size])

        #                                               MAIN LOOP                                                 #
        i = 0
        stop = 0
        initialization()
        while True:  # for i in range(max_iter)
            w_i = inertia_weight_update(i, max_iter)
            cost = objective_function(self.g_best_pos, target_pos)
            # If the cost is greater than the minimal error
            if cost > min_error:
                # Update the velocities and position.
                for part in range(swarm_size):
                    # Update velocity
                    cognitive = (influence['c1'] * np.random.uniform(0, 1, [1, num_seg * 2])) * (
                            self.p_best_pos[part, :] - self.current_pos[part, :])
                    social = (influence['c2'] * np.random.uniform(0, 1, [1, num_seg * 2])) * (
                            self.g_best_pos - self.current_pos[part, :])
                    self.velocity[part, :] = w_i * self.velocity[part, :] + cognitive + social
                    boundary_condition(part, 'v')
                    # Update position
                    self.current_pos[part, :] = self.current_pos[part, :] + self.velocity[part, :]
                    boundary_condition(part, 'x')
            else:
                # Return the parameters if the cost is less than the min_error
                final_params = self.g_best_pos
                return final_params
            for particle in range(swarm_size):
                # Set the personal best option
                cost_particle = objective_function(self.current_pos[particle, :], target_pos)  # cost_particle = error
                cost_p_best = objective_function(self.p_best_pos[particle, :], target_pos)
                if cost_particle < cost_p_best:  # if actual error is lower than the best personal error
                    self.p_best_pos[particle, :] = self.current_pos[particle, :]  # update new personal best position
                    # Set the global best option
                    cost_g_best = objective_function(self.g_best_pos, target_pos)
                    if cost_p_best < cost_g_best:  # if this particle has lower cost then global best
                        self.g_best_pos = self.current_pos[particle, :]

            i += 1
            if i == max_iter + 1:
                # If no good solution was found start again with new random position of particles,
                # this prevents from convergence to local minima
                initialization()
                i = 0
                stop += 1
                if stop == 21:
                    return np.array([0.0])
