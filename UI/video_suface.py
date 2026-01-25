# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'video_suface.ui'
##
## Created by: Qt User Interface Compiler version 6.6.3
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
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsView, QGridLayout,
    QHeaderView, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTextBrowser,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(886, 777)
        MainWindow.setToolTipDuration(6)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Plain)
        self.gridLayout_5 = QGridLayout(self.frame)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.frame_4 = QFrame(self.frame)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setMinimumSize(QSize(800, 300))
        self.frame_4.setFrameShape(QFrame.Box)
        self.frame_4.setFrameShadow(QFrame.Plain)
        self.gridLayout_6 = QGridLayout(self.frame_4)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.graphicsViewImage = QGraphicsView(self.frame_4)
        self.graphicsViewImage.setObjectName(u"graphicsViewImage")

        self.gridLayout_6.addWidget(self.graphicsViewImage, 0, 1, 1, 1)

        self.graphicsViewGrays = QGraphicsView(self.frame_4)
        self.graphicsViewGrays.setObjectName(u"graphicsViewGrays")

        self.gridLayout_6.addWidget(self.graphicsViewGrays, 0, 2, 1, 1)


        self.gridLayout_5.addWidget(self.frame_4, 0, 0, 2, 2)

        self.frame_3 = QFrame(self.frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.Box)
        self.frame_3.setFrameShadow(QFrame.Sunken)
        self.gridLayout_4 = QGridLayout(self.frame_3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.frame_2 = QFrame(self.frame_3)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMinimumSize(QSize(200, 200))
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(13)
        self.frame_2.setFont(font)
        self.frame_2.setFrameShape(QFrame.Box)
        self.frame_2.setFrameShadow(QFrame.Plain)
        self.gridLayout_2 = QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.history_button = QPushButton(self.frame_2)
        self.history_button.setObjectName(u"history_button")

        self.gridLayout_2.addWidget(self.history_button, 1, 0, 1, 1)

        self.log_button = QPushButton(self.frame_2)
        self.log_button.setObjectName(u"log_button")

        self.gridLayout_2.addWidget(self.log_button, 2, 0, 1, 1)

        self.exit_button = QPushButton(self.frame_2)
        self.exit_button.setObjectName(u"exit_button")

        self.gridLayout_2.addWidget(self.exit_button, 3, 0, 1, 1)

        self.config_button = QPushButton(self.frame_2)
        self.config_button.setObjectName(u"config_button")

        self.gridLayout_2.addWidget(self.config_button, 0, 0, 1, 1)


        self.gridLayout_4.addWidget(self.frame_2, 0, 3, 1, 1)

        self.textBrowser = QTextBrowser(self.frame_3)
        self.textBrowser.setObjectName(u"textBrowser")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setMinimumSize(QSize(600, 100))

        self.gridLayout_4.addWidget(self.textBrowser, 1, 0, 1, 4)

        self.face_display_label = QLabel(self.frame_3)
        self.face_display_label.setObjectName(u"face_display_label")
        self.face_display_label.setEnabled(True)
        sizePolicy.setHeightForWidth(self.face_display_label.sizePolicy().hasHeightForWidth())
        self.face_display_label.setSizePolicy(sizePolicy)
        self.face_display_label.setMinimumSize(QSize(200, 200))
        font1 = QFont()
        font1.setPointSize(20)
        self.face_display_label.setFont(font1)
        self.face_display_label.setFrameShape(QFrame.Box)
        self.face_display_label.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.face_display_label, 0, 2, 1, 1)

        self.history_tableWidget = QTableWidget(self.frame_3)
        if (self.history_tableWidget.columnCount() < 2):
            self.history_tableWidget.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.history_tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.history_tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        self.history_tableWidget.setObjectName(u"history_tableWidget")
        sizePolicy.setHeightForWidth(self.history_tableWidget.sizePolicy().hasHeightForWidth())
        self.history_tableWidget.setSizePolicy(sizePolicy)
        self.history_tableWidget.setMinimumSize(QSize(400, 200))
        font2 = QFont()
        font2.setPointSize(13)
        self.history_tableWidget.setFont(font2)

        self.gridLayout_4.addWidget(self.history_tableWidget, 0, 0, 1, 2)


        self.gridLayout_5.addWidget(self.frame_3, 3, 0, 1, 2)


        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u4eba\u4f53\u5065\u5eb7\u89c6\u9891\u76d1\u6d4b\u8f6f\u4ef6", None))
        self.history_button.setText(QCoreApplication.translate("MainWindow", u"\u5386\u53f2\u6570\u636e", None))
        self.log_button.setText(QCoreApplication.translate("MainWindow", u"\u65e5\u5fd7\u67e5\u770b", None))
        self.exit_button.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa\u7a0b\u5e8f", None))
        self.config_button.setText(QCoreApplication.translate("MainWindow", u"\u53c2\u6570\u8bbe\u7f6e", None))
        self.face_display_label.setText(QCoreApplication.translate("MainWindow", u"\u6293\u62cd", None))
        ___qtablewidgetitem = self.history_tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"\u2014\u2014\u62a5\u8b66\u65f6\u95f4\u2014\u2014", None));
        ___qtablewidgetitem1 = self.history_tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"\u2014\u2014\u62a5\u8b66\u6e29\u5ea6\u2014\u2014", None));
    # retranslateUi

