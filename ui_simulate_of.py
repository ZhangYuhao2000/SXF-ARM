# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simulate.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtGui import *
import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5 import QtCore, QtGui, QtWidgets


class Myfigure(FigureCanvas):
    def __init__(self):
        self.fig = plt.figure(figsize=(4, 4))  # 可选参数,facecolor为背景颜色facecolor='#FFD7C4',
        FigureCanvas.__init__(self, self.fig)  # 初始化激活widget中的plt部分
        self.axes = Axes3D(self.fig)

    def _print(self):
        # self.axes.mouse_init()
        self.draw()


class Ui_Form_SIMULATE(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(532, 672)

        self.figure = Myfigure()
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.figure)
        self.centralwidget = QtWidgets.QWidget(Form)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setLayout(self.vlayout)
        self.widget.setGeometry(QtCore.QRect(20, 240, 501, 391))
        self.widget.setObjectName("matplotlib")

        self.figure.axes.set_xlabel('X')
        self.figure.axes.set_ylabel('Y')
        self.figure.axes.set_zlabel('Z')
        self.figure.axes.set_xlim(300, -300)  # X轴，横向向右方向
        self.figure.axes.set_ylim(300, -300)  # Y轴,左向与X,Z轴互为垂直
        self.figure.axes.set_zlim(-100, 500)  # 竖向为Z轴

        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(90, 60, 91, 21))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 10, 111, 41))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(20, 60, 61, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 90, 91, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(20, 90, 61, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(90, 120, 91, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setGeometry(QtCore.QRect(20, 120, 61, 21))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(90, 150, 91, 21))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit_4 = QtWidgets.QLineEdit(Form)
        self.lineEdit_4.setGeometry(QtCore.QRect(20, 150, 61, 21))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(90, 180, 91, 21))
        self.pushButton_5.setObjectName("pushButton_5")
        self.lineEdit_5 = QtWidgets.QLineEdit(Form)
        self.lineEdit_5.setGeometry(QtCore.QRect(20, 180, 61, 21))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(90, 210, 91, 21))
        self.pushButton_6.setObjectName("pushButton_6")
        self.lineEdit_6 = QtWidgets.QLineEdit(Form)
        self.lineEdit_6.setGeometry(QtCore.QRect(20, 210, 61, 21))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(430, 640, 93, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.horizontalSlider = QtWidgets.QSlider(Form)
        self.horizontalSlider.setGeometry(QtCore.QRect(200, 60, 160, 22))
        self.horizontalSlider.setMinimum(-90)
        self.horizontalSlider.setMaximum(90)
        self.horizontalSlider.setProperty("value", 0)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(370, 60, 71, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(370, 90, 71, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(370, 120, 71, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(370, 150, 71, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(370, 180, 71, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(370, 210, 71, 21))
        self.label_7.setObjectName("label_7")
        self.horizontalSlider_2 = QtWidgets.QSlider(Form)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(200, 90, 160, 22))
        self.horizontalSlider_2.setMinimum(-90)
        self.horizontalSlider_2.setMaximum(90)
        self.horizontalSlider_2.setProperty("value", 0)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalSlider_3 = QtWidgets.QSlider(Form)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(200, 120, 160, 22))
        self.horizontalSlider_3.setMinimum(-90)
        self.horizontalSlider_3.setMaximum(90)
        self.horizontalSlider_3.setProperty("value", 0)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalSlider_4 = QtWidgets.QSlider(Form)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(200, 150, 160, 22))
        self.horizontalSlider_4.setMinimum(-90)
        self.horizontalSlider_4.setMaximum(90)
        self.horizontalSlider_4.setProperty("value", 0)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.horizontalSlider_5 = QtWidgets.QSlider(Form)
        self.horizontalSlider_5.setGeometry(QtCore.QRect(200, 180, 160, 22))
        self.horizontalSlider_5.setMinimum(-90)
        self.horizontalSlider_5.setMaximum(90)
        self.horizontalSlider_5.setProperty("value", 0)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.horizontalSlider_6 = QtWidgets.QSlider(Form)
        self.horizontalSlider_6.setGeometry(QtCore.QRect(200, 210, 160, 22))
        self.horizontalSlider_6.setMinimum(-90)
        self.horizontalSlider_6.setMaximum(90)
        self.horizontalSlider_6.setProperty("value", 0)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "关节 1"))
        self.label.setText(_translate("Form", "仿真器"))
        self.pushButton_2.setText(_translate("Form", "关节 2"))
        self.pushButton_3.setText(_translate("Form", "关节 3"))
        self.pushButton_4.setText(_translate("Form", "关节 4"))
        self.pushButton_5.setText(_translate("Form", "关节 5"))
        self.pushButton_6.setText(_translate("Form", "关节 6"))
        self.pushButton_7.setText(_translate("Form", "关 闭"))
        self.label_2.setText(_translate("Form", "0"))
        self.label_3.setText(_translate("Form", "0"))
        self.label_4.setText(_translate("Form", "0"))
        self.label_5.setText(_translate("Form", "0"))
        self.label_6.setText(_translate("Form", "0"))
        self.label_7.setText(_translate("Form", "0"))
        self.lineEdit.setText("0")
        self.lineEdit_2.setText("0")
        self.lineEdit_3.setText("0")
        self.lineEdit_4.setText("0")
        self.lineEdit_5.setText("0")
        self.lineEdit_6.setText("0")
