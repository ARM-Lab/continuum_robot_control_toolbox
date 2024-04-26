import json
import numpy as np


def piecewise_cc(num_seg: int,
                 theta: np.ndarray[float],
                 phi: np.ndarray[float],
                 seg_len: np.ndarray[float],
                 di: float,
                 num_of_el: np.ndarray[int],
                 optimizer=False) -> np.ndarray[float]:
    """
    Function creates transformation map using state "q" parametrization.

    :param num_seg: Number of segments.
    :param theta: Segment bending angle [deg].
    :param phi: Segment bending plane rotation angle [rad].
    :param seg_len: Segment lengths [m].
    :param di: Arc end connection distance from origin of local coordinate system [m].
    :param num_of_el: Number of elements per segment if n=1 all segments with equal number of points.
    :param optimizer: Set to True if function is used in PSO, function than returns only (Ex, Ey, Ez).

    :return g : Backbone curve with m 4x4 transformation matrices, where m is total number of points, reshaped
    into 1x16 vector (column-wise).
    """

    def tf_matrix_computation():
        """
        Function calculates the transformation matrix.

        :return rot: Transformation matrix.
        """
        delta = np.linalg.norm(np.array([del_x, del_y]))
        theta_q = delta / di

        rot = np.array([
            [1 + np.power((del_x / delta), 2) * (np.cos(theta_q) - 1),
             (del_x * del_y) / np.power(delta, 2) * (np.cos(theta_q) - 1), (del_x / delta) * np.sin(theta_q), 0],
            [(del_x * del_y) / np.power(delta, 2) * (np.cos(theta_q) - 1),
             1 + np.power((del_y / delta), 2) * (np.cos(theta_q) - 1), (del_y / delta) * np.sin(theta_q), 0],
            [-(del_x / delta) * np.sin(theta_q), -(del_y / delta) * np.sin(theta_q), np.cos(theta_q), 0],
            [0, 0, 0, 0]
        ])  # , dtype="object"
        # Modifies the last column of rot to include translation of elements in matrix
        t_temp = [del_x * (1 - np.cos(theta_q)), del_y * (1 - np.cos(theta_q)), delta * np.sin(theta_q)]
        t_trans = [k * ((di * li) / np.power(delta, 2)) for k in t_temp]
        t_trans.append(1)
        rot[:, 3] = t_trans
        return rot

    # Control if input arrays have same shape
    if theta.shape != phi.shape or theta.shape != seg_len.shape:
        raise ValueError("Dimension mismatch.")

    if num_of_el.size == 1 and num_seg > 1:  # If 1 parameter in vect->num_of_el and num_of_seg > 1
        num_of_el = np.tile(num_of_el, num_seg)  # Create an array that is num_of_el long with the num_seg repeated

    g = np.zeros((np.sum(num_of_el), 16))  # Stores the transformation matrices of all the points
    # in all the segments as rows
    base = np.eye(4)
    counter = 0  # Cycle counter
    li = del_x = del_y = None

    for i in range(num_seg):  # For each segment
        li = seg_len[i]
        del_x = theta[i] * seg_len[i] * di * np.cos(phi[i])
        del_y = theta[i] * seg_len[i] * di * np.sin(phi[i])

        if not optimizer:
            for j in range(1, num_of_el[i] + 1):  # Iterates over the elements within the segment
                li = seg_len[i]/num_of_el[i]*j
                del_x = theta[i] * j * seg_len[i] / num_of_el[i] * di * np.cos(phi[i])
                del_y = theta[i] * j * seg_len[i] / num_of_el[i] * di * np.sin(phi[i])
                if j != 0 and theta[i] != 0:  # TF matrix for every element excluding base element
                    tf_matrix = tf_matrix_computation()
                else:  # base element and avoiding of division by 0 when theta[i] is zero
                    tf_matrix = np.eye(4)
                    tf_matrix[:, 3] = [0, 0, j * (seg_len[i] / num_of_el[i]), 1]
                # Column-wise reshape
                g[counter, :] = (base @ tf_matrix).T.reshape((1, 16))
                # base @ rot: Performs a matrix multiplication
                # .T attribute transposes the resulting matrix
                # .reshape((1, 16)): Reshapes the transposed matrix into a 1x16 matrix
                # Updates the transformation matrix for the current point in the g array
                counter += 1
            base = g[counter - 1, :].reshape(4, 4).T  # last-most point's transformation matrix is the new base
        else:
            if theta[i] != 0:
                tf_matrix = tf_matrix_computation()
            else:  # base element and avoiding of division by 0 when theta[i] is zero
                tf_matrix = np.eye(4)
                tf_matrix[:, 3] = [0, 0, seg_len[i], 1]
            base @= tf_matrix

    if optimizer:
        return base[:3, -1]
    else:
        return g


def update_data(robot_parameters):
    """
    Function creates piecewise_cc_data.json file with stored g (Transformation matrices).

    :param robot_parameters: g, 1x16 transformation matrices.
    """
    data_for_json = robot_parameters.tolist()  # .json cannot store array data type

    with open("piecewise_cc_data.json", mode="w") as data_file:
        json.dump(data_for_json, data_file, indent=4)


def actuator_space_mapping(num_tendons: int,
                           num_of_el: np.ndarray[int],
                           seg_len: np.ndarray[float],
                           di: float,
                           kinematics: str,
                           partial_path=False,
                           **kwargs):
    """
    Function calculates parameters between actuator and configuration space based on selected algorithm

    :param num_tendons: Number of tendons.
    :param num_of_el: Number of elements per segment.
    :param seg_len: Segment lengths.
    :param di: Arc end connection distance from origin of local coordinate system [m].
    :param kinematics: "f" (for forward) or "tendon_lengths" (for inverse), determines if return are configuration space
     parameters or actuator space changes in tendon lengths.
    :param partial_path: If true kinematics with partially constrained tendons is considered.
    :param kwargs: When forward kinematics, lengths = (3 x seg_len) np.ndarray with tendon lengths changes, when inverse
    kinematics kwargs are theta and phi (-180 to 180°), all np.arrays.

    :return: Parameters based on selected algorithm.
    """
    result_list = []
    iteration = 0
    #                                      Forward robot-specific kinematics                                      #
    if kinematics == "f":
        def angle_computation(len_of_tendons):
            segment_length = seg_len[iteration]
            ac1 = len_of_tendons[0]  # lengths on the first actuator for tendon_lengths-th segment
            ac2 = len_of_tendons[1]
            ac3 = len_of_tendons[2]
            if num_tendons == 3:
                u_ = (ac2 - ac3) / (np.sqrt(3) * di)
                v_ = (segment_length - ac1) / di
            elif num_tendons == 4:
                ac4 = len_of_tendons[3]
                u_ = (ac2 - ac4) / (2*di)
                v_ = (ac3 - ac1) / (2*di)
            else:
                u_ = v_ = None
            theta_ = np.rad2deg(np.linalg.norm([u_, v_]) / segment_length)
            if theta_ != 0:  # to avoid division by 0
                if v_ != 0:
                    phi_ = np.rad2deg(np.arctan(-u_ / v_))
                    # print(f"phi {phi}")
                    if phi_ < 0:
                        phi_ += 180
                else:
                    phi_ = 90
            else:
                phi_ = 0.0
            if v_ < 0:
                phi_ += 180
            phi_ = round(phi_, 6)
            theta_ = round(theta_, 6)
            return phi_, theta_
        # FORWARD KINEMATICS MAINLOOP
        segments = kwargs["lengths"] + np.tile(seg_len, (num_tendons, 1)).T  # full tendon length
        for tendon_lengths in segments:
            temp = angle_computation(tendon_lengths)
            phi = temp[0]
            theta = temp[1]
            result_list.append([phi, theta])

            if partial_path and theta != 0:
                # converts tendon considered as circular arc to it's partially constrained equivalent.
                new_result_list = []
                theta = np.deg2rad(theta)
                for tendon in tendon_lengths:
                    new_result = (tendon * theta) / (2 * num_of_el[iteration] * np.sin(
                        theta / (2 * num_of_el[iteration])))
                    new_result_list.append(new_result)
                partial_angles = angle_computation(new_result_list)
                result_list[iteration] = partial_angles
            iteration += 1
        return np.array(result_list)
    #                                      Inverse robot-specific kinematics                                      #
    elif kinematics == "i":
        for theta, phi, l in zip(kwargs["theta"], kwargs["phi"], seg_len):
            theta = np.deg2rad(theta)
            phi = np.deg2rad(phi)
            h = l
            if theta != 0:  # to avoid division by 0
                v = np.sqrt((np.power((theta*h), 2) / (np.power(np.tan(phi), 2) + 1)))
                u = - np.tan(phi) * v
            else:
                v = u = 0
            if phi >= np.pi:  # opposite position for phi in range (180-360(0))
                u *= -1
                v *= -1
            if num_tendons == 3:
                ac_1 = h - di * v
                ac_2 = h + (1 / 2) * di * (v + np.sqrt(3) * u)
                ac_3 = h + (1 / 2) * di * (v - np.sqrt(3) * u)
                result_list.append([ac_1, ac_2, ac_3])
            elif num_tendons == 4:
                ac_1 = h - di * v
                ac_2 = h + di * u
                ac_3 = h + di * v
                ac_4 = h - di * u
                result_list.append([ac_1, ac_2, ac_3, ac_4])
            if partial_path and theta != 0:
                # converts tendon considered as circular arc to it's partially constrained equivalent.
                new_result_list = []
                for tendon in result_list[iteration]:
                    new_result = (tendon / theta) * 2 * num_of_el[iteration] * np.sin(
                        theta / (2 * num_of_el[iteration]))
                    new_result_list.append(new_result)
                result_list[iteration] = new_result_list
            iteration += 1
        solution = np.array(result_list) - np.tile(seg_len, (num_tendons, 1)).T
        return solution
