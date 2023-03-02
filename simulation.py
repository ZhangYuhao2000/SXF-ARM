import sys
from math import cos
from math import sin
import serial
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, QColor, QSize
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ui_simulate_of import *


simulation_joint1 = 0
simulation_joint2 = 0
simulation_joint3 = 0
simulation_joint4 = 0
simulation_joint5 = 0
simulation_joint6 = 0
output = np.zeros((249, 3))

pi = 3.1415926
# 机械臂参数
d1 = 50
a3 = 200
d3 = -30
a4 = 60
d4 = 150
d7 = 100


class Simulate:
    # plt.figure()
    # ax = plt.subplot(111, projection='3d')
    #
    # def __init__(self):
    #     self.ax.set_xlim(500, -500)  # X轴，横向向右方向
    #     self.ax.set_ylim(500, -500)  # Y轴,左向与X,Z轴互为垂直
    #     self.ax.set_zlim(-500, 500)  # 竖向为Z轴
    #     self.ax.set_xlabel('x')
    #     self.ax.set_ylabel('y')
    #     self.ax.set_zlabel('z')

    def t_mul(self, t1, t2):
        t1_1 = np.array(t1)
        t2_1 = np.array(t2)
        t = np.dot(t1_1, t2_1)
        return t

    def t_tran(self, theta, d, a, alpha):
        t = [[cos(theta), -sin(theta), 0, a],
             [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
             [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
             [0, 0, 0, 1]]
        return t

    def set_point(self, com):
        fp = 10
        point = np.ones((30, com), dtype=float)
        for x in range(0, 10):
            point[x][0] = x*fp
            point[x][1] = 0
            point[x][2] = 0

        for x in range(10, 20):
            point[x][0] = 0
            point[x][1] = (x - 10)*fp
            point[x][2] = 0

        for x in range(20, 30):
            point[x][0] = 0
            point[x][1] = 0
            point[x][2] = (x - 20)*fp

        return point

    def draw_point(self, point_input):
        p_xx = point_input[0:10, 0]
        p_xy = point_input[0:10, 1]
        p_xz = point_input[0:10, 2]
        self.ax.plot3D(p_xx, p_xy, p_xz, 'red')  # x轴
        p_yx = point_input[10:20, 0]
        p_yy = point_input[10:20, 1]
        p_yz = point_input[10:20, 2]
        self.ax.plot3D(p_yx, p_yy, p_yz, 'green')  # y轴
        p_zx = point_input[20:30, 0]
        p_zy = point_input[20:30, 1]
        p_zz = point_input[20:30, 2]
        self.ax.plot3D(p_zx, p_zy, p_zz, 'blue')  # z轴

    def point_calculate(self, t_r, point_input):
        i, j = point_input.shape
        for i in range(0, i):
            pr = np.dot(t_r, point_input.T)
            pr_c = pr.T[:, 0:3]
        return pr_c

    def coordinate_visualization(self, theta1, theta2, theta3, theta4, theta5, theta6):
        global pi
        global d1
        global a3
        global d3
        global a4
        global d4
        global d7
        theta = [(theta1*pi)/180, (theta2*pi)/180, (theta3*pi)/180, (theta4*pi)/180, (theta5*pi)/180, (theta6*pi)/180, 0]
        d = [d1, 0, d3, d4, 0, 0, d7]
        a = [0, 0, a3, a4, 0, 0, 0]
        alpha = [0, -pi/2, 0, -pi/2, pi/2, -pi/2, 0]

        t01 = self.t_tran(theta[0], d[0], a[0], alpha[0])
        t12 = self.t_tran(theta[1], d[1], a[1], alpha[1])
        t23 = self.t_tran(theta[2], d[2], a[2], alpha[2])
        t34 = self.t_tran(theta[3], d[3], a[3], alpha[3])
        t45 = self.t_tran(theta[4], d[4], a[4], alpha[4])
        t56 = self.t_tran(theta[5], d[5], a[5], alpha[5])
        t6t = self.t_tran(theta[6], d[6], a[6], alpha[6])

        tps1 = self.t_tran(theta[2], 0, a3, alpha[2])
        tps2 = self.t_tran(theta[3], 0, a4, alpha[3])

        point0 = self.set_point(3)
        # self.draw_point(point0)

        point1 = self.set_point(4)
        point1 = self.point_calculate(t01, point1)
        # self.draw_point(point1)

        point2 = self.set_point(4)
        point2 = self.point_calculate(self.t_mul(t01, t12), point2)
        # self.draw_point(point2)

        point3 = self.set_point(4)
        point3 = self.point_calculate(self.t_mul(self.t_mul(t01, t12), t23), point3)
        # self.draw_point(point3)

        point4 = self.set_point(4)
        point4 = self.point_calculate(self.t_mul(self.t_mul(self.t_mul(t01, t12), t23), t34), point4)
        # self.draw_point(point4)

        point5 = self.set_point(4)
        point5 = self.point_calculate(self.t_mul(self.t_mul(self.t_mul(self.t_mul(t01, t12), t23), t34), t45), point5)
        # self.draw_point(point5)

        point6 = self.set_point(4)
        point6 = self.point_calculate(self.t_mul(self.t_mul(self.t_mul(self.t_mul(self.t_mul(t01, t12), t23), t34), t45), t56), point6)
        # self.draw_point(point6)

        pointt = self.set_point(4)
        pointt = self.point_calculate(self.t_mul(self.t_mul(self.t_mul(self.t_mul(self.t_mul(self.t_mul(t01, t12), t23), t34), t45), t56), t6t), pointt)
        # self.draw_point(pointt)

        point_tps1 = np.zeros((1, 4), dtype=float)
        point_tps1[0][3] = 1
        point_tps1 = self.point_calculate(self.t_mul(self.t_mul(t01, t12), tps1), point_tps1)
        point_tps2 = np.zeros((1, 4), dtype=float)
        point_tps2[0][3] = 1
        point_tps2 = self.point_calculate(self.t_mul(self.t_mul(self.t_mul(t01, t12), t23), tps2), point_tps2)

        point_r = np.ones((9, 3), dtype=float)
        point_r[0] = point1[0]
        point_r[1] = point2[0]
        point_r[2] = point_tps1
        point_r[3] = point3[0]
        point_r[4] = point_tps2
        point_r[5] = point4[0]
        point_r[6] = point5[0]
        point_r[7] = point6[0]
        point_r[8] = pointt[0]
        # p_xx = point_r[0:9, 0]
        # p_xy = point_r[0:9, 1]
        # p_xz = point_r[0:9, 2]
        # self.ax.plot3D(p_xx, p_xy, p_xz, 'gray')
        #
        # plt.show()
        point_y = np.concatenate((point0, point1, point2, point3, point4, point5, point6, pointt, point_r), axis=0)

        return point_y, t01, t12, t23, t34, t45, t56, t6t


class arm_simulate(QWidget, Ui_Form_SIMULATE):
    def __init__(self):
        super(arm_simulate, self).__init__()
        self.setupUi(self)
        self.simulate = Simulate()

    def j1_input(self):
        global simulation_joint1
        get_value = int(self.lineEdit.text())
        self.horizontalSlider.setValue(get_value)
        self.label_2.setText(str(get_value))
        simulation_joint1 = get_value
        self.but_click()

    def j1_value_update_horizontalSlider(self):
        global simulation_joint1
        get_value = self.horizontalSlider.value()
        self.lineEdit.setText(str(get_value))
        self.label_2.setText(str(get_value))
        simulation_joint1 = get_value

    def j1_value_update_horizontalSlider_u(self):
        global simulation_joint1
        get_value = self.horizontalSlider.value()
        self.lineEdit.setText(str(get_value))
        self.label_2.setText(str(get_value))
        simulation_joint1 = get_value
        self.but_click()

    def j2_input(self):
        global simulation_joint2
        get_value = int(self.lineEdit_2.text())
        self.horizontalSlider_2.setValue(get_value)
        self.label_3.setText(str(get_value))
        simulation_joint2 = get_value
        self.but_click()

    def j2_value_update_horizontalSlider(self):
        global simulation_joint2
        get_value = self.horizontalSlider_2.value()
        self.lineEdit_2.setText(str(get_value))
        self.label_3.setText(str(get_value))
        simulation_joint2 = get_value

    def j2_value_update_horizontalSlider_u(self):
        global simulation_joint2
        get_value = self.horizontalSlider_2.value()
        self.lineEdit_2.setText(str(get_value))
        self.label_3.setText(str(get_value))
        simulation_joint2 = get_value
        self.but_click()

    def j3_input(self):
        global simulation_joint3
        get_value = int(self.lineEdit_3.text())
        self.horizontalSlider_3.setValue(get_value)
        self.label_4.setText(str(get_value))
        simulation_joint3 = get_value
        self.but_click()

    def j3_value_update_horizontalSlider(self):
        global simulation_joint3
        get_value = self.horizontalSlider_3.value()
        self.lineEdit_3.setText(str(get_value))
        self.label_4.setText(str(get_value))
        simulation_joint3 = get_value

    def j3_value_update_horizontalSlider_u(self):
        global simulation_joint3
        get_value = self.horizontalSlider_3.value()
        self.lineEdit_3.setText(str(get_value))
        self.label_4.setText(str(get_value))
        simulation_joint3 = get_value
        self.but_click()

    def j4_input(self):
        global simulation_joint4
        get_value = int(self.lineEdit_4.text())
        self.horizontalSlider_4.setValue(get_value)
        self.label_5.setText(str(get_value))
        simulation_joint4 = get_value
        self.but_click()

    def j4_value_update_horizontalSlider(self):
        global simulation_joint4
        get_value = self.horizontalSlider_4.value()
        self.lineEdit_4.setText(str(get_value))
        self.label_5.setText(str(get_value))
        simulation_joint4 = get_value

    def j4_value_update_horizontalSlider_u(self):
        global simulation_joint4
        get_value = self.horizontalSlider_4.value()
        self.lineEdit_4.setText(str(get_value))
        self.label_5.setText(str(get_value))
        simulation_joint4 = get_value
        self.but_click()

    def j5_input(self):
        global simulation_joint5
        get_value = int(self.lineEdit_5.text())
        self.horizontalSlider_5.setValue(get_value)
        self.label_6.setText(str(get_value))
        simulation_joint5 = get_value
        self.but_click()

    def j5_value_update_horizontalSlider(self):
        global simulation_joint5
        get_value = self.horizontalSlider_5.value()
        self.lineEdit_5.setText(str(get_value))
        self.label_6.setText(str(get_value))
        simulation_joint5 = get_value

    def j5_value_update_horizontalSlider_u(self):
        global simulation_joint5
        get_value = self.horizontalSlider_5.value()
        self.lineEdit_5.setText(str(get_value))
        self.label_6.setText(str(get_value))
        simulation_joint5 = get_value
        self.but_click()

    def j6_input(self):
        global simulation_joint6
        get_value = int(self.lineEdit_6.text())
        self.horizontalSlider_6.setValue(get_value)
        self.label_7.setText(str(get_value))
        simulation_joint6 = get_value
        self.but_click()

    def j6_value_update_horizontalSlider(self):
        global simulation_joint6
        get_value = self.horizontalSlider_6.value()
        self.lineEdit_6.setText(str(get_value))
        self.label_7.setText(str(get_value))
        simulation_joint6 = get_value

    def j6_value_update_horizontalSlider_u(self):
        global simulation_joint6
        get_value = self.horizontalSlider_6.value()
        self.lineEdit_6.setText(str(get_value))
        self.label_7.setText(str(get_value))
        simulation_joint6 = get_value
        self.but_click()



    def but_refresh(self):
        self.figure.axes.cla()
        self.figure.axes._init_axis()
        self.figure.axes.set_xlabel('X')
        self.figure.axes.set_ylabel('Y')
        self.figure.axes.set_zlabel('Z')
        self.figure.axes.set_xlim(300, -300)  # X轴，横向向右方向
        self.figure.axes.set_ylim(300, -300)  # Y轴,左向与X,Z轴互为垂直
        self.figure.axes.set_zlim(-100, 500)  # 竖向为Z轴
        self.figure.draw()

    def but_click(self):
        global pi
        global d1
        global a3
        global d3
        global a4
        global d4
        global d7

        global simulation_joint1
        global simulation_joint2
        global simulation_joint3
        global simulation_joint4
        global simulation_joint5
        global simulation_joint6
        global output
        output, t1, t2, t3, t4, t5, t6, t7 = self.simulate.coordinate_visualization(simulation_joint1,
                                                                                    simulation_joint2-90,
                                                                                    simulation_joint3,
                                                                                    simulation_joint4,
                                                                                    simulation_joint5,
                                                                                    simulation_joint6)
        self.but_refresh()
        p = 0
        for i in range(0, 24):
            if p == 0:
                self.figure.axes.plot(output[i*10:(i+1)*10-1, 0], output[i*10:(i+1)*10-1, 1], output[i*10:(i+1)*10-1, 2], 'red')  # x轴
                p = p + 1
            elif p == 1:
                self.figure.axes.plot(output[i*10:(i + 1)*10 - 1, 0], output[i*10:(i + 1)*10 - 1, 1], output[i*10:(i + 1)*10 - 1, 2], 'green')  # y轴
                p = p + 1
            elif p == 2:
                self.figure.axes.plot(output[i*10:(i + 1)*10 - 1, 0], output[i*10:(i + 1)*10 - 1, 1], output[i*10:(i + 1)*10 - 1, 2], 'blue')  # z轴
                p = 0
        self.figure.axes.plot(output[240:249, 0], output[240:249, 1], output[240:249, 2], 'gray')
        point_vertical = np.zeros((2, 3))
        point_vertical[0][0] = output[248][0]
        point_vertical[0][1] = output[248][1]
        point_vertical[0][2] = output[248][2]
        point_vertical[1][0] = output[248][0]
        point_vertical[1][1] = output[248][1]
        point_vertical[1][2] = 0
        self.figure.axes.plot(point_vertical[:, 0], point_vertical[:, 1], point_vertical[:, 2], 'orange')
        t_str = "(" + str(round(point_vertical[0][0], 2)) + "," + str(round(point_vertical[0][1], 2)) + "," + str(round(point_vertical[0][2], 2)) + ")"
        self.figure.axes.text(point_vertical[0][0], point_vertical[0][1], point_vertical[0][2], t_str, color='blue')

        #  视觉美化部分代码 底座
        point_clc_up = np.zeros((9, 3))
        point_clc_down = np.zeros((9, 3))
        point_clc_con = np.zeros((9, 2, 3))
        clc_r = 50
        ang = 0
        for i in range(0, 9):
            ang = pi*2*i/8
            point_clc_up[i][0] = clc_r*cos(ang)  # X值
            point_clc_up[i][1] = clc_r*sin(ang)   # Y值
            point_clc_up[i][2] = d1
            point_clc_down[i][0] = clc_r*cos(ang)  # X值
            point_clc_down[i][1] = clc_r*sin(ang)  # Y值
            point_clc_con[i][0][0] = clc_r*cos(ang)
            point_clc_con[i][0][1] = clc_r*sin(ang)
            point_clc_con[i][0][2] = 0
            point_clc_con[i][1][0] = clc_r*cos(ang)
            point_clc_con[i][1][1] = clc_r*sin(ang)
            point_clc_con[i][1][2] = d1
        self.figure.axes.plot(point_clc_up[:, 0], point_clc_up[:, 1], point_clc_up[:, 2], 'gray')
        self.figure.axes.plot(point_clc_down[:, 0], point_clc_down[:, 1], point_clc_down[:, 2], 'gray')
        for i in range(0, 9):
            self.figure.axes.plot(point_clc_con[i, :, 0], point_clc_con[i, :, 1], point_clc_con[i, :, 2], 'gray')

        #  视觉美化部分代码 臂1
        arm_w_1 = 20
        point_j1_up = np.ones((5, 4))
        point_j1_down = np.ones((5, 4))
        point_j1_line1 = np.ones((2, 4))
        point_j1_line2 = np.ones((2, 4))
        point_j1_line3 = np.ones((2, 4))
        point_j1_line4 = np.ones((2, 4))

        point_j1_up[0][2] = arm_w_1
        point_j1_up[0][1] = arm_w_1
        point_j1_up[0][0] = a3
        point_j1_up[1][2] = arm_w_1
        point_j1_up[1][1] = -arm_w_1
        point_j1_up[1][0] = a3
        point_j1_up[2][2] = -arm_w_1
        point_j1_up[2][1] = -arm_w_1
        point_j1_up[2][0] = a3
        point_j1_up[3][2] = -arm_w_1
        point_j1_up[3][1] = arm_w_1
        point_j1_up[3][0] = a3
        point_j1_up[4][2] = arm_w_1
        point_j1_up[4][1] = arm_w_1
        point_j1_up[4][0] = a3

        point_j1_down[0][2] = arm_w_1
        point_j1_down[0][1] = arm_w_1
        point_j1_down[0][0] = 0
        point_j1_down[1][2] = arm_w_1
        point_j1_down[1][1] = -arm_w_1
        point_j1_down[1][0] = 0
        point_j1_down[2][2] = -arm_w_1
        point_j1_down[2][1] = -arm_w_1
        point_j1_down[2][0] = 0
        point_j1_down[3][2] = -arm_w_1
        point_j1_down[3][1] = arm_w_1
        point_j1_down[3][0] = 0
        point_j1_down[4][2] = arm_w_1
        point_j1_down[4][1] = arm_w_1
        point_j1_down[4][0] = 0

        point_j1_line1[0][2] = arm_w_1
        point_j1_line1[0][1] = arm_w_1
        point_j1_line1[0][0] = a3
        point_j1_line1[1][2] = arm_w_1
        point_j1_line1[1][1] = arm_w_1
        point_j1_line1[1][0] = 0
        point_j1_line2[0][2] = arm_w_1
        point_j1_line2[0][1] = -arm_w_1
        point_j1_line2[0][0] = a3
        point_j1_line2[1][2] = arm_w_1
        point_j1_line2[1][1] = -arm_w_1
        point_j1_line2[1][0] = 0
        point_j1_line3[0][2] = -arm_w_1
        point_j1_line3[0][1] = -arm_w_1
        point_j1_line3[0][0] = a3
        point_j1_line3[1][2] = -arm_w_1
        point_j1_line3[1][1] = -arm_w_1
        point_j1_line3[1][0] = 0
        point_j1_line4[0][2] = -arm_w_1
        point_j1_line4[0][1] = arm_w_1
        point_j1_line4[0][0] = a3
        point_j1_line4[1][2] = -arm_w_1
        point_j1_line4[1][1] = arm_w_1
        point_j1_line4[1][0] = 0

        t_g_1 = self.simulate.t_mul(t1, t2)
        point_j1_up = self.simulate.point_calculate(t_g_1, point_j1_up)
        point_j1_down = self.simulate.point_calculate(t_g_1, point_j1_down)
        point_j1_line1 = self.simulate.point_calculate(t_g_1, point_j1_line1)
        point_j1_line2 = self.simulate.point_calculate(t_g_1, point_j1_line2)
        point_j1_line3 = self.simulate.point_calculate(t_g_1, point_j1_line3)
        point_j1_line4 = self.simulate.point_calculate(t_g_1, point_j1_line4)

        self.figure.axes.plot(point_j1_up[:, 0], point_j1_up[:, 1], point_j1_up[:, 2], 'gray')
        self.figure.axes.plot(point_j1_down[:, 0], point_j1_down[:, 1], point_j1_down[:, 2], 'gray')
        self.figure.axes.plot(point_j1_line1[:, 0], point_j1_line1[:, 1], point_j1_line1[:, 2], 'gray')
        self.figure.axes.plot(point_j1_line2[:, 0], point_j1_line2[:, 1], point_j1_line2[:, 2], 'gray')
        self.figure.axes.plot(point_j1_line3[:, 0], point_j1_line3[:, 1], point_j1_line3[:, 2], 'gray')
        self.figure.axes.plot(point_j1_line4[:, 0], point_j1_line4[:, 1], point_j1_line4[:, 2], 'gray')

        #  视觉美化部分代码 臂2
        arm_w_2 = 15
        point_j2_up = np.ones((5, 4))
        point_j2_down = np.ones((5, 4))
        point_j2_line1 = np.ones((2, 4))
        point_j2_line2 = np.ones((2, 4))
        point_j2_line3 = np.ones((2, 4))
        point_j2_line4 = np.ones((2, 4))

        point_j2_up[0][0] = arm_w_2
        point_j2_up[0][1] = arm_w_2
        point_j2_up[0][2] = -d4
        point_j2_up[1][0] = arm_w_2
        point_j2_up[1][1] = -arm_w_2
        point_j2_up[1][2] = -d4
        point_j2_up[2][0] = -arm_w_2
        point_j2_up[2][1] = -arm_w_2
        point_j2_up[2][2] = -d4
        point_j2_up[3][0] = -arm_w_2
        point_j2_up[3][1] = arm_w_2
        point_j2_up[3][2] = -d4
        point_j2_up[4][0] = arm_w_2
        point_j2_up[4][1] = arm_w_2
        point_j2_up[4][2] = -d4

        point_j2_down[0][0] = arm_w_2
        point_j2_down[0][1] = arm_w_2
        point_j2_down[0][2] = 0
        point_j2_down[1][0] = arm_w_2
        point_j2_down[1][1] = -arm_w_2
        point_j2_down[1][2] = 0
        point_j2_down[2][0] = -arm_w_2
        point_j2_down[2][1] = -arm_w_2
        point_j2_down[2][2] = 0
        point_j2_down[3][0] = -arm_w_2
        point_j2_down[3][1] = arm_w_2
        point_j2_down[3][2] = 0
        point_j2_down[4][0] = arm_w_2
        point_j2_down[4][1] = arm_w_2
        point_j2_down[4][2] = 0

        point_j2_line1[0][0] = arm_w_2
        point_j2_line1[0][1] = arm_w_2
        point_j2_line1[0][2] = -d4
        point_j2_line1[1][0] = arm_w_2
        point_j2_line1[1][1] = arm_w_2
        point_j2_line1[1][2] = 0
        point_j2_line2[0][0] = arm_w_2
        point_j2_line2[0][1] = -arm_w_2
        point_j2_line2[0][2] = -d4
        point_j2_line2[1][0] = arm_w_2
        point_j2_line2[1][1] = -arm_w_2
        point_j2_line2[1][2] = 0
        point_j2_line3[0][0] = -arm_w_2
        point_j2_line3[0][1] = -arm_w_2
        point_j2_line3[0][2] = -d4
        point_j2_line3[1][0] = -arm_w_2
        point_j2_line3[1][1] = -arm_w_2
        point_j2_line3[1][2] = 0
        point_j2_line4[0][0] = -arm_w_2
        point_j2_line4[0][1] = arm_w_2
        point_j2_line4[0][2] = -d4
        point_j2_line4[1][0] = -arm_w_2
        point_j2_line4[1][1] = arm_w_2
        point_j2_line4[1][2] = 0

        t_g_2 = self.simulate.t_mul(self.simulate.t_mul(self.simulate.t_mul(t1, t2), t3), t4)
        point_j2_up = self.simulate.point_calculate(t_g_2, point_j2_up)
        point_j2_down = self.simulate.point_calculate(t_g_2, point_j2_down)
        point_j2_line1 = self.simulate.point_calculate(t_g_2, point_j2_line1)
        point_j2_line2 = self.simulate.point_calculate(t_g_2, point_j2_line2)
        point_j2_line3 = self.simulate.point_calculate(t_g_2, point_j2_line3)
        point_j2_line4 = self.simulate.point_calculate(t_g_2, point_j2_line4)

        self.figure.axes.plot(point_j2_up[:, 0], point_j2_up[:, 1], point_j2_up[:, 2], 'gray')
        self.figure.axes.plot(point_j2_down[:, 0], point_j2_down[:, 1], point_j2_down[:, 2], 'gray')
        self.figure.axes.plot(point_j2_line1[:, 0], point_j2_line1[:, 1], point_j2_line1[:, 2], 'gray')
        self.figure.axes.plot(point_j2_line2[:, 0], point_j2_line2[:, 1], point_j2_line2[:, 2], 'gray')
        self.figure.axes.plot(point_j2_line3[:, 0], point_j2_line3[:, 1], point_j2_line3[:, 2], 'gray')
        self.figure.axes.plot(point_j2_line4[:, 0], point_j2_line4[:, 1], point_j2_line4[:, 2], 'gray')

        #  视觉美化部分代码 臂3
        arm_w_3 = 7
        point_j3_up = np.ones((5, 4))
        point_j3_down = np.ones((5, 4))
        point_j3_line1 = np.ones((2, 4))
        point_j3_line2 = np.ones((2, 4))
        point_j3_line3 = np.ones((2, 4))
        point_j3_line4 = np.ones((2, 4))

        point_j3_up[0][0] = arm_w_3
        point_j3_up[0][1] = arm_w_3
        point_j3_up[0][2] = d7
        point_j3_up[1][0] = arm_w_3
        point_j3_up[1][1] = -arm_w_3
        point_j3_up[1][2] = d7
        point_j3_up[2][0] = -arm_w_3
        point_j3_up[2][1] = -arm_w_3
        point_j3_up[2][2] = d7
        point_j3_up[3][0] = -arm_w_3
        point_j3_up[3][1] = arm_w_3
        point_j3_up[3][2] = d7
        point_j3_up[4][0] = arm_w_3
        point_j3_up[4][1] = arm_w_3
        point_j3_up[4][2] = d7

        point_j3_down[0][0] = arm_w_3
        point_j3_down[0][1] = arm_w_3
        point_j3_down[0][2] = 0
        point_j3_down[1][0] = arm_w_3
        point_j3_down[1][1] = -arm_w_3
        point_j3_down[1][2] = 0
        point_j3_down[2][0] = -arm_w_3
        point_j3_down[2][1] = -arm_w_3
        point_j3_down[2][2] = 0
        point_j3_down[3][0] = -arm_w_3
        point_j3_down[3][1] = arm_w_3
        point_j3_down[3][2] = 0
        point_j3_down[4][0] = arm_w_3
        point_j3_down[4][1] = arm_w_3
        point_j3_down[4][2] = 0

        point_j3_line1[0][0] = arm_w_3
        point_j3_line1[0][1] = arm_w_3
        point_j3_line1[0][2] = d7
        point_j3_line1[1][0] = arm_w_3
        point_j3_line1[1][1] = arm_w_3
        point_j3_line1[1][2] = 0
        point_j3_line2[0][0] = arm_w_3
        point_j3_line2[0][1] = -arm_w_3
        point_j3_line2[0][2] = d7
        point_j3_line2[1][0] = arm_w_3
        point_j3_line2[1][1] = -arm_w_3
        point_j3_line2[1][2] = 0
        point_j3_line3[0][0] = -arm_w_3
        point_j3_line3[0][1] = -arm_w_3
        point_j3_line3[0][2] = d7
        point_j3_line3[1][0] = -arm_w_3
        point_j3_line3[1][1] = -arm_w_3
        point_j3_line3[1][2] = 0
        point_j3_line4[0][0] = -arm_w_3
        point_j3_line4[0][1] = arm_w_3
        point_j3_line4[0][2] = d7
        point_j3_line4[1][0] = -arm_w_3
        point_j3_line4[1][1] = arm_w_3
        point_j3_line4[1][2] = 0

        t_g_3 = self.simulate.t_mul(self.simulate.t_mul(self.simulate.t_mul(self.simulate.t_mul(self.simulate.t_mul(t1, t2), t3), t4), t5), t6)
        point_j3_up = self.simulate.point_calculate(t_g_3, point_j3_up)
        point_j3_down = self.simulate.point_calculate(t_g_3, point_j3_down)
        point_j3_line1 = self.simulate.point_calculate(t_g_3, point_j3_line1)
        point_j3_line2 = self.simulate.point_calculate(t_g_3, point_j3_line2)
        point_j3_line3 = self.simulate.point_calculate(t_g_3, point_j3_line3)
        point_j3_line4 = self.simulate.point_calculate(t_g_3, point_j3_line4)

        self.figure.axes.plot(point_j3_up[:, 0], point_j3_up[:, 1], point_j3_up[:, 2], 'gray')
        self.figure.axes.plot(point_j3_down[:, 0], point_j3_down[:, 1], point_j3_down[:, 2], 'gray')
        self.figure.axes.plot(point_j3_line1[:, 0], point_j3_line1[:, 1], point_j3_line1[:, 2], 'gray')
        self.figure.axes.plot(point_j3_line2[:, 0], point_j3_line2[:, 1], point_j3_line2[:, 2], 'gray')
        self.figure.axes.plot(point_j3_line3[:, 0], point_j3_line3[:, 1], point_j3_line3[:, 2], 'gray')
        self.figure.axes.plot(point_j3_line4[:, 0], point_j3_line4[:, 1], point_j3_line4[:, 2], 'gray')

        self.figure._print()

    def quick(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    simulate = arm_simulate()

    simulate.horizontalSlider.sliderMoved.connect(simulate.j1_value_update_horizontalSlider)
    simulate.horizontalSlider.sliderReleased.connect(simulate.j1_value_update_horizontalSlider_u)
    simulate.horizontalSlider_2.sliderMoved.connect(simulate.j2_value_update_horizontalSlider)
    simulate.horizontalSlider_2.sliderReleased.connect(simulate.j2_value_update_horizontalSlider_u)
    simulate.horizontalSlider_3.sliderMoved.connect(simulate.j3_value_update_horizontalSlider)
    simulate.horizontalSlider_3.sliderReleased.connect(simulate.j3_value_update_horizontalSlider_u)
    simulate.horizontalSlider_4.sliderMoved.connect(simulate.j4_value_update_horizontalSlider)
    simulate.horizontalSlider_4.sliderReleased.connect(simulate.j4_value_update_horizontalSlider_u)
    simulate.horizontalSlider_5.sliderMoved.connect(simulate.j5_value_update_horizontalSlider)
    simulate.horizontalSlider_5.sliderReleased.connect(simulate.j5_value_update_horizontalSlider_u)
    simulate.horizontalSlider_6.sliderMoved.connect(simulate.j6_value_update_horizontalSlider)
    simulate.horizontalSlider_6.sliderReleased.connect(simulate.j6_value_update_horizontalSlider_u)

    simulate.pushButton.clicked.connect(simulate.j1_input)
    simulate.pushButton_2.clicked.connect(simulate.j2_input)
    simulate.pushButton_3.clicked.connect(simulate.j3_input)
    simulate.pushButton_4.clicked.connect(simulate.j4_input)
    simulate.pushButton_5.clicked.connect(simulate.j5_input)
    simulate.pushButton_6.clicked.connect(simulate.j6_input)
    simulate.pushButton_7.clicked.connect(simulate.quick)
    simulate.show()

    sys.exit(app.exec_())
