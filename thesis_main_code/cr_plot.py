import numpy as np
import matplotlib.figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # Collection of 3D polygons


class PlotSetup:
    """
    Sets up the matplotlib figure.

    g: (start_position) n 4x4 backbone curve position transformation matrices reshaped
    into 1x16 vector (column-wise).
    new_g: (end_position) n 4x4 backbone curve position transformation matrices reshaped
     into 1x16 vector (column-wise).
    end_index: Indices of where tdcr segments ends.
    """
    def __init__(self, g: np.ndarray[float], new_g: np.ndarray[float], end_index: np.ndarray[int]):
        self.g = g
        self.new_g = new_g
        self.end_index = end_index
        self.g_rot = [[self.g[value-1, 0:3], self.g[value-1, 4:7], self.g[value-1, 8:11]] for value in self.end_index]
        # list of 3 arrays for X, Y, Z rotation
        self.new_g_rot = None
        self.actual_g = self.g[:, 12:15]
        self.actual_rot = self.g_rot

        self.fig = matplotlib.figure.Figure(figsize=(5, 5), label='TDCR')
        self.ax = self.fig.add_subplot(projection='3d')

    def __call__(self, i):
        num_frames = 30
        self.ax.clear()
        if i == 0:
            self.new_g_rot = [[self.new_g[value-1, 0:3], self.new_g[value-1, 4:7], self.new_g[value-1, 8:11]] for value in self.end_index]
            if len(self.g) != len(self.new_g):
                self.g = self.new_g
                self.actual_g = self.new_g
                self.g_rot = self.new_g_rot
                self.actual_rot = self.new_g_rot
            self.axes_setup()
            self.plot_backbone()
            self.coordinate_systems()
        else:
            self.actual_g = np.array(self.g[:, 12:15] + ((self.new_g[:, 12:15] - self.g[:, 12:15]) / (num_frames-1) * i))
            segment_list, rotation_list = [], []
            for segment in range(len(self.new_g_rot)):
                for arr in range(3):
                    rotation_list.append(np.array(self.g_rot[segment][arr][:] + ((self.new_g_rot[segment][arr][:] - self.g_rot[segment][arr][:]) / (num_frames-1) * i)))
                segment_list.append(rotation_list)
                rotation_list = []
            self.actual_rot = segment_list
            self.axes_setup()
            self.plot_backbone()
            self.coordinate_systems()
        self.ax.legend()

    def axes_setup(self):
        # Setting axes, labels and title
        backbone_z_length = np.sum(np.linalg.norm(self.actual_g[1:] - self.actual_g[:-1], axis=1))
        #  computes the curve length along the second axis "axis = 1"
        #  np.linalg.norm -> ord = "none" -> returns magnitude
        #  g[1:, 12:15] - g[:-1,12:15] calculates the element-wise difference between 2 points end point - start point
        clearance = 0.02  # This parameter create a visual clearance between the surface and other elements
        xmax = np.max(np.abs(self.actual_g[:, 0])) + clearance  # maximal value for X-axis
        ymax = np.max(np.abs(self.actual_g[:, 1])) + clearance
        zmax = backbone_z_length + clearance
        self.ax.set(xlim=(-xmax, xmax), ylim=(-ymax, ymax), zlim=(0, zmax),
                    xlabel='x [m]', ylabel='y [m]', zlabel='z [m]',
                    aspect='equal',
                    title='Tendon Driven Continuum Robot\nPosition and Orientation')

    def plot_backbone(self):
        # Backbone of robot
        s = 0
        for i in range(len(self.end_index)):
            rgb_val = (1 / len(self.end_index)) * i
            self.ax.plot(self.actual_g[s:self.end_index[i], 0],
                         self.actual_g[s:self.end_index[i], 1],
                         self.actual_g[s:self.end_index[i], 2],
                         color=(rgb_val, rgb_val, rgb_val), label=f'segment {i + 1}', lw=2)
            s = self.end_index[i]
    
    def coordinate_systems(self):
        # Coordinate systems
        colors = ['FireBrick', 'DarkGreen', 'DarkBlue']
        axes = ['x', 'y', 'z']
        fake_line = [(0, 0, 0), (0, 0, 0)]
        fake_line_list = [fake_line.copy() for _ in range(3)]

        def coordinate_sys(start, end, arrow_index: [int]):
            """
            Nested function for coordinate system creation
            start: 1x3 (x, y, z) start point coordinates
            end: 1x3 (u, v, w)
            arrow_index: (0-2) index based color picking
            """
            self.ax.quiver(start[0], start[1], start[2], end[0], end[1], end[2], color=colors[arrow_index], length=0.01)

        # Segment frame coordinate systems
        cycle_counter = 0
        for value in self.end_index:
            x, y, z = self.actual_g[value - 1, 0], self.actual_g[value - 1, 1], self.actual_g[value - 1, 2]
            for i in range(3):
                rot_x, rot_y, rot_z = self.actual_rot[cycle_counter][i][0], \
                    self.actual_rot[cycle_counter][i][1], \
                    self.actual_rot[cycle_counter][i][2]
                coordinate_sys((x, y, z), (rot_x, rot_y, rot_z), i)
                if cycle_counter == 0:  # this is for legend
                    self.ax.add_collection3d(Line3DCollection(fake_line_list, colors=colors[i], label=axes[i]))
            cycle_counter += 1

        # Base frame coordinate system
        s_point = np.zeros((3, 1))
        end_point = np.eye(3)
        for j in range(3):
            coordinate_sys(s_point, end_point[j], j)

