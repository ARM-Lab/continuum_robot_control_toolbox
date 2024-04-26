import numpy as np
from scipy import optimize
import pyswarms as ps

#angles = np.zeros(4)
#lengths = np.array([0.5, 0.4, 0.3, 0.2])


def forward_kin(angle_list: np.array, length_list: np.array):
    TF = np.eye(4)
    for i, j in zip(angle_list, length_list):
        t_temp = np.array([[np.cos(i), -np.sin(i), 0, j*np.cos(i)],
                           [np.sin(i), np.cos(i), 0, j*np.sin(i)],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        TF @= t_temp
    return TF


def solver(q0, l):
    qMin = -20
    qMax = 20
    bounds = ((qMin, qMax), (qMin, qMax), (qMin, qMax), (qMin, qMax))
    result = optimize.minimize(objective_function, q0, l, method='SLSQP', bounds=bounds)
    for k in range(4):
        print(result.x[k])


def objective_function(q0, l):
    x_real = forward_kin(q0, l)
    x_des = np.array(([1, 0, 0, 0.8],
                      [0, 1, 0, 0.5],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]))
    error = np.linalg.norm(x_des - x_real)
    return error


def main():
        print('Spustam...')
        angles = np.zeros(4)
        lengths = np.array([0.5, 0.4, 0.3, 0.2])
        solver(angles, lengths)


if __name__ == "__main__":
    main()
    angles = np.array([0.498611084736275, 1.357973002345483, -2.2787058999225147, 0.4220708105091806])
    lengths = np.array([0.5, 0.4, 0.3, 0.2])
    f = forward_kin(angles, lengths)
    print(f'x= {f[0, -1]}, y= {f[1,-1]}')
