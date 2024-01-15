# la parte gráfica del código fue generada con PyQt5 y después transformada
# de .ui a .py.

from PyQt5 import QtCore, QtWidgets
from functools import partial
# para permitirnos llamar una función con args desde button.clicked.connect
from music21 import stream, note, midi
import random
from main import Melody_Generator
from main import chords_seed, reverse_mapping, length, L_symb
import tensorflow as tf
import keras

model = keras.models.load_model("saved_model_200_piano.keras")
Melody_Generator(model, chords_seed, reverse_mapping, length, L_symb, 100, [], 1)

keyDetune = []
keyDetune2 = []
a = stream.Stream()  # aquí se guarda el stream de teclas tocadas


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        # define la pantalla
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(626, 357)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Botón A
        self.Button_A = QtWidgets.QPushButton(self.centralwidget)
        self.Button_A.setGeometry(QtCore.QRect(40, 80, 75, 231))
        self.Button_A.setStyleSheet("background-color: rgb(255, 255, 0);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_A.setObjectName("Button_A")
        self.Button_A.clicked.connect(partial(self.button_click, self.Button_A))

        # Botón B
        self.Button_B = QtWidgets.QPushButton(self.centralwidget)
        self.Button_B.setGeometry(QtCore.QRect(120, 80, 75, 231))
        self.Button_B.setStyleSheet("background-color: rgb(0, 255, 0);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_B.setObjectName("Button_B")
        self.Button_B.clicked.connect(partial(self.button_click, self.Button_B))

        # Botón C
        self.Button_C = QtWidgets.QPushButton(self.centralwidget)
        self.Button_C.setGeometry(QtCore.QRect(200, 80, 75, 231))
        self.Button_C.setStyleSheet("background-color: rgb(255, 0, 127);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_C.setObjectName("Button_C")
        self.Button_C.clicked.connect(partial(self.button_click, self.Button_C))

        # Botón D
        self.Button_D = QtWidgets.QPushButton(self.centralwidget)
        self.Button_D.setGeometry(QtCore.QRect(280, 80, 75, 231))
        self.Button_D.setStyleSheet("background-color: rgb(255, 85, 0);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_D.setObjectName("Button_D")
        self.Button_D.clicked.connect(partial(self.button_click, self.Button_D))

        # Botón E
        self.Button_E = QtWidgets.QPushButton(self.centralwidget)
        self.Button_E.setGeometry(QtCore.QRect(360, 80, 75, 231))
        self.Button_E.setStyleSheet("\n"
                                    "background-color: rgb(170, 0, 255);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_E.setObjectName("Button_E")
        self.Button_E.clicked.connect(partial(self.button_click, self.Button_E))

        # Botón F
        self.Button_F = QtWidgets.QPushButton(self.centralwidget)
        self.Button_F.setGeometry(QtCore.QRect(440, 80, 75, 231))
        self.Button_F.setStyleSheet("background-color: rgb(170, 0, 0);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_F.setObjectName("Button_F")
        self.Button_F.clicked.connect(partial(self.button_click, self.Button_F))

        # Botón G
        self.Button_G = QtWidgets.QPushButton(self.centralwidget)
        self.Button_G.setGeometry(QtCore.QRect(520, 80, 75, 231))
        self.Button_G.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                    "font: 26pt \"MS Gothic\";")
        self.Button_G.setObjectName("Button_G")
        self.Button_G.clicked.connect(partial(self.button_click, self.Button_G))

        # Botón label (PIANO)
        self.label = QtWidgets.QPushButton(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 10, 551, 51))
        self.label.setStyleSheet("background-color: rgb(0, 170, 127);\n"
                                 "font: 26pt \"MS Gothic\";")
        self.label.setObjectName("label")
        self.label.clicked.connect(partial(self.button_recall, self.label))

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 626, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Piano"))
        self.Button_A.setText(_translate("MainWindow", "A"))
        self.Button_B.setText(_translate("MainWindow", "B"))
        self.Button_C.setText(_translate("MainWindow", "C"))
        self.Button_D.setText(_translate("MainWindow", "D"))
        self.Button_E.setText(_translate("MainWindow", "E"))
        self.Button_F.setText(_translate("MainWindow", "F"))
        self.Button_G.setText(_translate("MainWindow", "G"))
        self.label.setText(_translate("MainWindow", "PIANO"))

    # Dependiendo del botón presionado se crea y reproduce
    # la nota correspondiente
    def button_click(self, button):
        if(button == self.Button_A):
            x = "A"
        if(button == self.Button_B):
            x = "B"
        if(button == self.Button_C):
            x = "C"
        if(button == self.Button_D):
            x = "D"
        if(button == self.Button_E):
            x = "E"
        if(button == self.Button_F):
            x = "F"
        if(button == self.Button_G):
            x = "G"

        new_note = note.Note(x)
        new_note.duration.quarterLength = 2
        b = stream.Stream()
        b.append(new_note)

        # el a.append es para guardar globalmente todo lo que ha sido tocado
        new_note2 = note.Note(x)
        a.append(new_note2)

        # Código sacado de
        # http://web.mit.edu/music21/doc/moduleReference/moduleMidiRealtime.html
        # permite reproducir el sonido instantaneamente
        for i in range(0, 127):
            keyDetune.append(random.randint(-30, 30))

        for n in b.flatten().notes:
            n.pitch.microtone = keyDetune[n.pitch.midi]

        sp = midi.realtime.StreamPlayer(b)
        sp.play()

    def button_recall(self, button):
        # permite reproducir el stream a, stream global
        # que guarda todo lo que ha sido tocado
        for i in range(0, 127):
            keyDetune2.append(random.randint(-30, 30))

        for n in a.flatten().notes:
            n.pitch.microtone = keyDetune2[n.pitch.midi]

        sp = midi.realtime.StreamPlayer(a)
        sp.play()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
