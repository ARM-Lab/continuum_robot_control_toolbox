import numpy as np
from pso_algorithm import ParticleSwarmOptimization
from forward_kinematics import piecewise_cc


def reachable_ep(target: np.array, num_seg: int, seg_len: np.array, num_of_el: np.array, di: float):

    # Finding a random solution for IK
    pso_object = ParticleSwarmOptimization()
    result = pso_object.optimize(num_seg=num_seg,
                                 seg_len=seg_len,
                                 num_of_el=num_of_el,
                                 di=di,
                                 target_pos=target)
    # Calculation of reachable EP poses of section 1.
    j = True
    phi_1 = result[num_seg]
    theta_1 = result[0]
    while j:
        # p1_specific = pos_1
        theta_1 += 1
        pos_1 = piecewise_cc(num_seg=1,
                             theta=theta_1,
                             phi=phi_1,
                             seg_len=seg_len[0],
                             di=di,
                             num_of_el=num_of_el[0],
                             optimizer=True)
        pos2_exists = # function()
        # TODO: Check the existence of pos_2 using PSO
        if not pos2_exists:
            j = False


# TODO select algorithm based on how many sections user selected
reachable_ep(target=np.array([0.008943704370872637, 0.0, 0.022721292000777805]),
             num_seg=1,
             seg_len=np.array([0.025]),
             num_of_el=np.array([10]),
             di=0.005)
