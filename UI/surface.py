# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'surface.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QTextBrowser, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1357, 974)
        font = QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.gridLayout_4 = QGridLayout(self.frame)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.frame_3 = QFrame(self.frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.Shape.Box)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.rgb_display_label = QLabel(self.frame_3)
        self.rgb_display_label.setObjectName(u"rgb_display_label")
        self.rgb_display_label.setMinimumSize(QSize(600, 300))
        self.rgb_display_label.setFrameShape(QFrame.Shape.Box)
        self.rgb_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.rgb_display_label, 0, 0, 1, 2)

        self.textBrowser = QTextBrowser(self.frame_3)
        self.textBrowser.setObjectName(u"textBrowser")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)

        self.gridLayout_3.addWidget(self.textBrowser, 1, 1, 1, 1)

        self.temp_display_label = QLabel(self.frame_3)
        self.temp_display_label.setObjectName(u"temp_display_label")
        self.temp_display_label.setMinimumSize(QSize(600, 300))
        self.temp_display_label.setFrameShape(QFrame.Shape.Box)
        self.temp_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.temp_display_label, 0, 2, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.star_button = QPushButton(self.frame_3)
        self.star_button.setObjectName(u"star_button")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.star_button)

        self.end_button = QPushButton(self.frame_3)
        self.end_button.setObjectName(u"end_button")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.end_button.sizePolicy().hasHeightForWidth())
        self.end_button.setSizePolicy(sizePolicy1)

        self.formLayout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.end_button)

        self.tmp_checkBox = QCheckBox(self.frame_3)
        self.tmp_checkBox.setObjectName(u"tmp_checkBox")
        self.tmp_checkBox.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.tmp_checkBox.setChecked(True)
        self.tmp_checkBox.setAutoRepeat(False)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.tmp_checkBox)

        self.face_checkbox = QCheckBox(self.frame_3)
        self.face_checkbox.setObjectName(u"face_checkbox")
        self.face_checkbox.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.face_checkbox.setChecked(True)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.face_checkbox)

        self.label = QLabel(self.frame_3)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label)

        self.label_2 = QLabel(self.frame_3)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.temp_bais_doubleSpinBox = QDoubleSpinBox(self.frame_3)
        self.temp_bais_doubleSpinBox.setObjectName(u"temp_bais_doubleSpinBox")
        sizePolicy1.setHeightForWidth(self.temp_bais_doubleSpinBox.sizePolicy().hasHeightForWidth())
        self.temp_bais_doubleSpinBox.setSizePolicy(sizePolicy1)
        self.temp_bais_doubleSpinBox.setMinimum(-100.000000000000000)

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.temp_bais_doubleSpinBox)

        self.warning_temp = QDoubleSpinBox(self.frame_3)
        self.warning_temp.setObjectName(u"warning_temp")
        sizePolicy1.setHeightForWidth(self.warning_temp.sizePolicy().hasHeightForWidth())
        self.warning_temp.setSizePolicy(sizePolicy1)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.warning_temp)

        self.label_3 = QLabel(self.frame_3)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.label_3)

        self.label_4 = QLabel(self.frame_3)
        self.label_4.setObjectName(u"label_4")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.label_4)


        self.gridLayout_3.addLayout(self.formLayout, 1, 0, 1, 1)


        self.gridLayout_4.addWidget(self.frame_3, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1357, 27))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.rgb_display_label.setText(QCoreApplication.translate("MainWindow", u"\u5f69\u8272\u56fe\u50cf\u5c55\u793a", None))
        self.temp_display_label.setText(QCoreApplication.translate("MainWindow", u"\u6e29\u5ea6\u4fe1\u606f\u5c55\u793a", None))
        self.star_button.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u6444\u50cf\u5934", None))
        self.end_button.setText(QCoreApplication.translate("MainWindow", u"\u5173\u95ed\u6444\u50cf\u5934", None))
        self.tmp_checkBox.setText("")
        self.face_checkbox.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u6e29\u5ea6\u8865\u507f", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u62a5\u8b66\u6e29\u5ea6l", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u6e29\u5ea6\u62a5\u8b66", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u4eba\u8138\u68c0\u6d4b", None))
    # retranslateUi

