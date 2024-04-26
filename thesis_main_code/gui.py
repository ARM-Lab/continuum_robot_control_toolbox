import os
import sys
import json
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from cr_plot import PlotSetup
from forward_kinematics import piecewise_cc, update_data, actuator_space_mapping
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pso_algorithm import ParticleSwarmOptimization
#                                                   VARIABLES                                                    #
FONT_NAME = 'Montserrat'
BG_COLOR = '#C1C1C1'
data_dict = {  # Start values
             'seg_len': [0.025, 0.020, 0.030],
             'num_of_el': [10],
             'phi': [0, 120, 240],
             'theta': [30, 15, 45],
             'end_index': [10, 20, 30],
             'num_seg': [3],
             'num_tend': [3],
             'theta_limit': [90],
             'phi_limit': [360],
             'di': [0.003]
            }
ani = ""


#                                                   FUNCTIONS                                                  #
def restart_fcn():  # Restart button
    """Function re-runs the program"""
    root.window.destroy()
    python = sys.executable
    os.execl(python, python, *sys.argv)


def data_calculator():
    """Function calculates forward kinematics and returns TF matrices"""
    g_matrices = piecewise_cc(num_seg=data_dict['num_seg'][0],
                              theta=np.array(data_dict['theta']),
                              phi=np.deg2rad(np.array(data_dict['phi'])),  # radians
                              di=data_dict['di'][0],  # scientific format
                              seg_len=np.array(data_dict['seg_len']),  # scientific format
                              num_of_el=np.array(data_dict['num_of_el']))
    update_data(robot_parameters=g_matrices)  # data update


def data_selector():
    """Function calculates TF-matrices and store them into .json file"""
    with open("./piecewise_cc_data.json", "r") as n:
        old = json.load(n)
    data_calculator()
    with open("./piecewise_cc_data.json", "r") as p:
        new = json.load(p)
    return old, new


# Animation
def update_animate():
    num_frames = 30

    def update_fcn(frame):
        global ani
        if frame < num_frames:
            ani = root.window.after(100, update_fcn, frame + 1)
            plot(frame)
            root.canvas.draw()
        else:
            root.window.after_cancel(ani)

    data_set = data_selector()
    plot.g = np.array(data_set[0])
    plot.new_g = np.array(data_set[1])
    plot.end_index = np.array(data_dict['end_index'])
    update_fcn(frame=0)


# Value checking
def value_checker(values, float_val=False):
    try:
        for value in values:
            if not float_val:
                int(value.get())
            else:
                float(value.get())
    except ValueError:
        if not float_val:
            messagebox.showerror('Value Error', 'ValueError: All entries have to be integer values.')
        else:
            messagebox.showerror('Value Error', 'ValueError: Float value expected.')
        return False
    else:
        return True


# Help button action
def help_b():
    messagebox.showinfo(title='Help ?',
                        message='1) To get started, enter the number of segments, tendons, tendon '
                                'connection distance, and robot angle limits. Than choose the desired kinematics and '
                                'tendon path. After that, click on the confirmation button.\n\n2) Next, fill in'
                                'the parameters for each segment, such as length, number of spacer '
                                'disks, bending plane angle (phi), and segment bending angle (theta)'
                                '(only for forward kinematics), with inverse kinematics type in coordinates [mm] of the'
                                ' target. To add data to the table, click on '
                                'the "Add" button.\n\n3) For the plot of the continuum robot, press'
                                ' the "plot TDCR" button.\n\n4) To modify the position entry parameters'
                                ', click on the "New position entry button".\n\nIMPORTANT NOTICE: '
                                'Please note that only whole numbers without any decimal points, '
                                'also known as integer values, are acceptable for submission, in all entries excluding '
                                'the target coordinates entries.')


class ContinuumRobotGUI:
    def __init__(self, robot_plot):
        self.robot_plot = robot_plot
        self.algorithm_selector = False  # Changing algorithm in function of the add button
        self.active_segment = 0  # Index of active segment
        self.table_data = []  # Table data manipulation
        self.ik_target = None
        self.kinematics = None

        # Window setup
        self.window = tk.Tk()
        self.window.state('zoomed')
        self.window.title('Continuum Robot GUI')
        self.window.config(bg=BG_COLOR, padx=20, pady=20)

        # DESIGN RECTANGLES
        a = tk.Canvas(width=360, height=100, highlightthickness=0)
        a.grid(column=2, row=3, columnspan=2, rowspan=2, sticky='E')
        a.create_text(95, 12, text='Kinematics selection:', font=(FONT_NAME, 10, 'bold'))
        a.create_text(265, 12, text='Tendon path selection:', font=(FONT_NAME, 10, 'bold'))
        a.create_line(180, 20, 180, 80, fill=BG_COLOR, width=2)

        b = tk.Canvas(width=666, height=2, highlightthickness=0)
        b.grid(column=0, row=5, columnspan=4, sticky='SE')

        c = tk.Canvas(width=2, height=606, highlightthickness=0)
        c.grid(column=4, row=0, rowspan=13, padx=(20, 0))

        d = tk.Canvas(width=500, height=160, highlightthickness=0)
        d.grid(column=1, row=6, columnspan=3, rowspan=4, pady=(5, 0), sticky='NE')
        d.create_text(60, 17, text='Configuration space\nvariables [°]:', font=(FONT_NAME, 9, 'bold'))
        d.create_text(320, 10, text='End-point position [mm]:', font=(FONT_NAME, 9, 'bold'))
        d.create_line(250, 20, 250, 130, fill=BG_COLOR, width=2)

        # Canvas setup
        self.canvas = FigureCanvasTkAgg(figure=robot_plot.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=5, row=0, rowspan=11, padx=(20, 0), sticky='S')

        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(column=5, row=11, sticky='NW', padx=(20, 0))

        # Label
        self.introduction = tk.Label(text='TDCR SETUP:', bg=BG_COLOR, font=(FONT_NAME, 18, 'bold'))
        self.introduction.grid(column=0, row=0, pady=5, sticky='N', columnspan=4)

        self.s_box_1 = tk.Label(text='Number of segments:', bg=BG_COLOR, font=(FONT_NAME, 10, 'bold'))
        self.s_box_1.grid(column=0, row=1, pady=(15, 5))

        self.s_box_2 = tk.Label(text='Number of tendons:', bg=BG_COLOR, font=(FONT_NAME, 10, 'bold'))
        self.s_box_2.grid(column=0, row=2, pady=(5, 10), padx=(10, 0))

        self.theta_limit = tk.Label(text='Angle limit - Theta [°]:', bg=BG_COLOR, font=(FONT_NAME, 10))
        self.theta_limit.grid(column=0, row=3, padx=(5, 0), sticky='S')

        self.phi_limit = tk.Label(text='Angle limit - Phi [°]:', bg=BG_COLOR, font=(FONT_NAME, 10))
        self.phi_limit.grid(column=1, row=3, padx=(0, 5), sticky='SW')

        self.di = tk.Label(text='Tendon connection distance\nd [mm]:', bg=BG_COLOR, font=(FONT_NAME, 10))
        self.di.grid(column=2, row=1, pady=(15, 0), padx=(0, 5))

        self.len_of_seg = tk.Label(text='Segment length [mm]:', bg=BG_COLOR, font=(FONT_NAME, 10))
        self.len_of_seg.grid(column=0, row=6, pady=(10, 5), padx=(5, 0), sticky='S')

        self.num_of_el = tk.Label(text='Number of elements:', bg=BG_COLOR, font=(FONT_NAME, 10))
        self.num_of_el.grid(column=0, row=8, pady=(10, 5), padx=(5, 0), sticky='S')

        lab_list = ['X', 'Y', 'Z']
        for char in lab_list:
            self.ep_lab = tk.Label(text=f'{char}:', font=(FONT_NAME, 9, 'bold'))
            self.ep_lab.grid(column=3, row=6 + lab_list.index(char), padx=(45, 0), sticky='SW')

        # SPINBOX
        # Number of segments spinbox
        self.spinbox1 = tk.Spinbox(from_=1, to=4, state='readonly', width=12, font=(FONT_NAME, 10),
                                   justify='center', borderwidth=2)
        self.spinbox1.grid(column=1, row=1, pady=(15, 5), padx=(0, 10), sticky='W')
        # Number of tendons spinbox
        self.spinbox2 = tk.Spinbox(from_=3, to=4, state='readonly', width=12, font=(FONT_NAME, 10),
                                   justify='center', borderwidth=2)
        self.spinbox2.grid(column=1, row=2, pady=(5, 10), padx=(0, 10), sticky='W')

        # BUTTON
        # Confirm setup button
        self.confirm_b = tk.Button(text='Confirm ✔', command=self.confirmation, width=20, cursor='hand2',
                                   font=(FONT_NAME, 10))
        self.confirm_b.grid(column=1, row=5, columnspan=2, pady=(10, 5), padx=(50, 0))
        # Add button
        self.add_b = tk.Button(text='Add ⬇', command=self.add_b_pressed, width=10, cursor='hand2', font=(FONT_NAME, 10),
                               state='disabled')
        self.add_b.grid(column=3, row=9, pady=5, padx=(0, 5), sticky='E')
        # Help button
        self.help_b = tk.Button(text='Help ?', command=help_b, width=10, bg='#E3EFF7', cursor='hand2',
                                font=(FONT_NAME, 10))
        self.help_b.grid(column=3, row=1, pady=(15, 5), sticky='E')
        # Plot TDCR button
        self.plot_b = tk.Button(text='Plot TDCR', command=self.plot_b, width=20, cursor='hand2', font=(FONT_NAME, 10),
                                state='disabled')
        self.plot_b.grid(column=5, row=12, pady=5)
        # New entry button
        self.new_entry_b = tk.Button(text='New position entry ⬆', command=self.new_entry_b, width=20, cursor='hand2',
                                     font=(FONT_NAME, 10), state='disabled')
        self.new_entry_b.grid(column=3, row=12, pady=5, sticky='E')
        # Actuator space variables button
        self.ac_button = tk.Button(text='Actuator space variables 🔎︎', command=self.ac_b, width=25, cursor='hand2',
                                   font=(FONT_NAME, 10), state='disabled')
        self.ac_button.grid(column=1, row=9, columnspan=2, padx=(15, 0), sticky='W')
        # Restart button
        self.restart = tk.Button(text='Restart ⟲', command=restart_fcn, width=12, cursor='hand2',
                                 font=(FONT_NAME, 10))
        self.restart.grid(column=5, row=12, pady=5, sticky='SE')

        # ENTRY
        self.entries = []
        for i in range(8):
            entry = tk.Entry(width=14, justify='center', font=(FONT_NAME, 10))
            if i > 2:
                entry.config(state='disabled')
            self.entries.append(entry)

        self.entries[0].insert(tk.END, string="90")
        self.entries[0].grid(column=0, row=4, padx=(5, 0))
        self.entries[1].insert(tk.END, string="360")
        self.entries[1].grid(column=1, row=4, padx=(0, 5), sticky='W')
        self.entries[2].insert(tk.END, string="3")
        self.entries[2].grid(column=2, row=2, pady=5, padx=(0, 10))
        self.entries[3].grid(column=0, row=7, padx=(5, 0), sticky='N')
        self.entries[4].grid(column=0, row=9, padx=(5, 0), sticky='N')
        self.entries[5].grid(column=3, row=6, padx=(0, 20), sticky='SE')
        self.entries[6].grid(column=3, row=7, padx=(0, 20), sticky='SE')
        self.entries[7].grid(column=3, row=8, padx=(0, 20), sticky='SE')

        # TREEVIEW
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.table = ttk.Treeview(self.window, style='My.Treeview', columns=('0', '1', '2', '3', '4'), show='headings',
                                  height=5, selectmode='none')
        self.table.column('0', anchor='center', stretch=False, width=100)
        self.table.heading('0', text='Segment number')
        self.table.column('1', anchor='center', stretch=False, width=125)
        self.table.heading('1', text='Segment length [mm]')  # convert
        self.table.column('2', anchor='center', stretch=False, width=120)
        self.table.heading('2', text='Number of elements')
        self.table.column('3', anchor='center', stretch=False, width=150)
        self.table.heading('3', text='Bending plane angle [deg]')
        self.table.column('4', anchor='center', stretch=False, width=165)
        self.table.heading('4', text='Segment bending angle [deg]')
        self.table.grid(column=0, row=10, rowspan=2, columnspan=4, pady=5, padx=(5, 0), sticky='W')

        # RADIOBUTTON
        self.radio_state = tk.StringVar(None, 'i')  # Default selection
        self.radiobuttonF = tk.Radiobutton(text='Forward kinematics', value='f', variable=self.radio_state,
                                           font=(FONT_NAME, 10), cursor='hand2')
        self.radiobuttonF.grid(column=2, row=3, sticky='S')
        self.radiobuttonI = tk.Radiobutton(text='Inverse kinematics', value='i', variable=self.radio_state,
                                           font=(FONT_NAME, 10), cursor='hand2')
        self.radiobuttonI.grid(column=2, row=4, padx=(0, 5), sticky='N')

        # LISTBOX
        self.listbox = tk.Listbox(height=2, font=(FONT_NAME, 10), cursor='hand2', width=18)
        self.listbox.insert(0, "fully constrained")
        self.listbox.insert(1, "partially constrained")
        self.listbox.grid(column=3, row=3, rowspan=2, pady=(0, 5))
        self.listbox.select_set(0)  # Default selection is fully constrained tendon path

        # SCALES
        self.scale_theta = tk.Scale(from_=0, to=data_dict['theta_limit'][0], font=(FONT_NAME, 8), cursor='hand2',
                                    orient='horizontal', resolution=1, state='disabled', width=10, sliderlength=10,
                                    length=110)
        self.scale_phi = tk.Scale(from_=0, to=data_dict['phi_limit'][0], font=(FONT_NAME, 8), cursor='hand2',
                                  orient='horizontal', resolution=1, state='disabled', width=10, sliderlength=10,
                                  length=110)
        self.scale_theta.grid(column=2, row=6, pady=(20, 0), sticky='W')
        self.scale_phi.grid(column=2, row=7, sticky='NW')
        theta = tk.Label(text=f'Theta:', font=(FONT_NAME, 10, 'bold'))
        theta.grid(column=1, row=6, pady=(5, 0), sticky='SE')
        phi = tk.Label(text=f'Phi:', font=(FONT_NAME, 10, 'bold'))
        phi.grid(column=1, row=7, pady=(5, 0), sticky='SE')

        # Closing message
        self.window.protocol('WM_DELETE_WINDOW', self.on_closing)

#                                                CLASS METHODS                                                #
    # Closing messagebox
    def on_closing(self):
        if messagebox.askokcancel('Quit', 'Do you want to quit?'):
            self.window.destroy()

    # Actuator space variables button
    def ac_b(self):
        ac_result = actuator_space_mapping(num_tendons=data_dict['num_tend'][0],
                                           num_of_el=np.array(data_dict['num_of_el']),
                                           di=data_dict['di'][0],
                                           kinematics="i",
                                           partial_path=self.listbox.index('active'),
                                           theta=np.array(data_dict['theta']),
                                           phi=np.array(data_dict['phi']),
                                           seg_len=np.array(data_dict['seg_len']))
        loop_variable = 1
        mes = ""
        for result in ac_result:
            nested_variable = 1
            mes += f"\nFor segment {loop_variable}.: "
            for length in result:
                mes += f"\nTendon {nested_variable}.: {round(length*1000, 4)} mm."
                nested_variable += 1
            loop_variable += 1

        message = f'You have chosen {self.spinbox1.get()} segment, {self.spinbox2.get()} tendon TDCR, with ' \
                  f'{self.listbox.get("active")}, tendon path.\n\nHere are tendon length changes: {mes}'
        messagebox.showinfo(title='Actuator space variables',
                            message=message)

    # Confirmation button pressed
    def confirmation(self):
        """Checks values in active entries, then disable all general setup entries
           and activate segment setup entries, also pre-fill them with values """
        self.kinematics = self.radio_state.get()
        if value_checker(values=self.entries[:3]):
            self.spinbox1.config(state='disabled')
            self.spinbox2.config(state='disabled')
            self.radiobuttonI.config(state='disabled')
            self.radiobuttonF.config(state='disabled')
            self.listbox.config(state='disabled')
            for x in self.entries[:3]:
                x.config(state='disabled')

            self.entries[3].config(state='normal')
            self.entries[3].insert(0, string=str(int(data_dict['seg_len'][0] * 1000)))
            self.entries[4].config(state='normal')
            self.entries[4].insert(0, string=str(data_dict['num_of_el'][0]))

            if self.kinematics == 'f':
                self.scale_theta.config(state='normal')
                self.scale_phi.config(state='normal')
                self.introduction.config(text=f'Segment {self.active_segment + 1} setup: ')
            elif self.kinematics == 'i':
                for cor in self.entries[5:]:
                    cor.config(state='normal')
                    self.introduction.config(text='Type end-point position (Ex, Ey, Ez) [mm].')
            self.add_b.config(state='normal')
            self.confirm_b.config(bg='MediumSeaGreen', state='disabled')

    # Add button pressed
    def add_b_pressed(self):
        """Checks values which go into table, based on algorithm_selector it appends all new data or only change data"""
        if not self.algorithm_selector:
            if value_checker(values=self.entries[3:5]):
                table_val = [self.active_segment + 1]
                for val in self.entries[3:5]:
                    table_val.append(val.get())
                if self.kinematics == 'f':
                    table_val.extend((self.scale_phi.get(), self.scale_theta.get()))
                else:
                    table_val.extend((0, 0))
                self.table.insert(parent='', index='end', values=table_val)
                self.active_segment += 1

                if self.active_segment == int(self.spinbox1.get()):
                    target_check = False
                    if self.kinematics == 'f':
                        self.scale_theta.config(state='disabled')
                        self.scale_phi.config(state='disabled')
                        self.active_segment = 0
                    elif value_checker(values=self.entries[5:], float_val=True):
                        target = []
                        self.active_segment = 0  # if everything is ok
                        for en in self.entries[5:]:
                            target.append(float(en.get()))
                            en.config(state='disabled')
                        self.ik_target = np.array(target) / 1000
                        target_check = True
                    else:
                        self.active_segment -= 1  # if kinematics is inverse and value_checker = False
                        children = self.table.get_children()
                        self.table.delete(children[-1])
                    if self.kinematics == 'f' or target_check:
                        self.introduction.config(text=f'TDCR setup completed !')
                        self.add_b.config(state='disabled')
                        self.plot_b.config(state='normal', bg='MediumSeaGreen')
                        for _ in self.entries[3:5]:
                            _.config(state='disabled')

                else:
                    self.introduction.config(text=f'Segment {self.active_segment + 1} setup: ')

        # Changing data in the table
        else:
            if self.kinematics == 'f':
                children = self.table.get_children()
                self.table_data[self.active_segment][-1] = self.scale_theta.get()
                self.table_data[self.active_segment][-2] = self.scale_phi.get()
                self.table.delete(children[self.active_segment])
                self.table.insert(parent='', index=self.active_segment, values=self.table_data[self.active_segment])
                self.active_segment += 1
                if self.active_segment == int(self.spinbox1.get()):
                    self.active_segment = 0
                    self.introduction.config(text=f'TDCR setup completed !')
                    self.scale_theta.config(state='disabled')
                    self.scale_phi.config(state='disabled')
                    self.ac_button.config(state='disabled', bg='#f0f0f0')
                    self.add_b.config(state='disabled')
                    self.plot_b.config(state='normal', bg='MediumSeaGreen')
                else:
                    self.introduction.config(text=f'Type configuration parameters for segment {self.active_segment + 1}')
            else:
                if value_checker(values=self.entries[5:], float_val=True):
                    target = []
                    for en in self.entries[5:]:
                        target.append(float(en.get()))
                        en.config(state='disabled')
                    self.ik_target = np.array(target) / 1000
                    self.ac_button.config(state='disabled', bg='#f0f0f0')
                    self.add_b.config(state='disabled')
                    self.plot_b.config(state='normal', bg='MediumSeaGreen')

    # Plot button pressed
    def plot_b(self):
        """Prepare data for plot"""
        self.table_data = []
        algorithm_stop = False
        self.plot_b.config(state='disabled', bg='#f0f0f0')
        children = self.table.get_children()
        for idn in children:
            self.table_data.append(self.table.item(idn)['values'])
        # Data preparation for plot
        for key in list(data_dict.keys()):
            del data_dict[key][::]
        data_dict['theta_limit'] += [int(self.entries[0].get())]
        data_dict['phi_limit'] += [int(self.entries[1].get())]
        data_dict['di'] += [int(self.entries[2].get()) / 1000]
        data_dict['num_seg'] += [int(self.spinbox1.get())]
        data_dict['num_tend'] += [int(self.spinbox2.get())]
        end_index = 0
        for _ in self.table_data:
            data_dict['seg_len'] += [int(_[1]) / 1000]
            data_dict['num_of_el'] += [int(_[2])]
            end_index += int(_[2])
            data_dict['end_index'] += [end_index]
            if self.kinematics == 'f':
                data_dict['phi'] += [int(_[3])]
                data_dict['theta'] += [int(_[4])]
        if self.kinematics == 'i':
            num_seg = data_dict['num_seg'][0]
            pso_object = ParticleSwarmOptimization()
            result = pso_object.optimize(num_seg=num_seg,
                                         seg_len=np.array(data_dict['seg_len']),
                                         num_of_el=np.array(data_dict['num_of_el']),
                                         di=data_dict['di'][0],
                                         angle_limits=np.array([data_dict['theta_limit'], data_dict['phi_limit']]),
                                         target_pos=self.ik_target)
            if result.size == 1:
                messagebox.showwarning(title='Solution not found',
                                       message="Inverse kinematic solver was not able to find a solution.\n"
                                               "Please ensure that the entered values are correct and inside the "
                                               "reachable workspace.")
                self.plot_b.config(state='disabled')
                algorithm_stop = True
            else:
                data_dict['theta'] = list(result[:num_seg])
                data_dict['phi'] = list(result[num_seg:])
                loop_var = 0
                for res_t, res_p in zip(result[:num_seg], result[num_seg:]):
                    self.table_data[loop_var][-1] = round(res_t, 2)
                    self.table_data[loop_var][-2] = round(res_p, 2)
                    children = self.table.get_children()
                    self.table.delete(children[loop_var])
                    self.table.insert(parent='', index=loop_var, values=self.table_data[loop_var])
                    loop_var += 1
                self.style.configure('Treeview.Heading', background='MediumSeaGreen')

        self.new_entry_b.config(state='normal')
        if not algorithm_stop:
            update_animate()
            self.ac_button.config(state='normal', bg='MediumSeaGreen')
            if self.kinematics == 'f':
                end_tip_position = np.round(plot.new_g[-1, 12:15]*1000, 2)
                loop_var = 0
                for coordinates in self.entries[5:]:
                    coordinates.config(state='normal', bg='MediumSeaGreen')
                    coordinates.insert(tk.END, string=f"{end_tip_position[loop_var]}")
                    loop_var += 1

    # New entry button pressed
    def new_entry_b(self):
        """Replaces all data in segment bending angle column with 0."""

        self.algorithm_selector = True  # add button now only actualize configuration space variables
        self.new_entry_b.config(state='disabled')
        for cor_entry in self.entries[5:]:
            cor_entry.delete(0, tk.END)
            cor_entry.config(state='normal')
            if self.kinematics == 'f':
                cor_entry.config(state='disabled')
        children = self.table.get_children()
        for item in children:
            self.table.delete(item)
        for data in self.table_data:
            data[-2] = 0
            data[-1] = 0
            self.table.insert(parent='', index='end', values=data)
        self.add_b.config(state='normal')
        if self.kinematics == 'f':
            self.scale_theta.config(state='normal')
            self.scale_phi.config(state='normal')
            self.introduction.config(text=f'Type new configuration parameters for segment {self.active_segment + 1}')
        else:
            self.introduction.config(text='Type new end-point position (Ex, Ey, Ez) [mm].')


if '__main__' == __name__:

    plot = PlotSetup(g=np.array(data_selector()[1]),
                     new_g=np.array([]),
                     end_index=np.array(data_dict['end_index']))

    root = ContinuumRobotGUI(robot_plot=plot)
    root.window.mainloop()
