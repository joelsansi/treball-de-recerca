import tkinter
import socket
from time import sleep
ventana = tkinter.Tk()
ventana.geometry("480x320")
label = tkinter.Label(ventana)


#-------------------- TCP SETUP ----------------------------------------------
host = "192.168.0.29"
port = 50000

s = socket.socket()
#-------------------FOR TCP SERVER ---------------------
print("host: " + host)
s.bind((host,port))
s.listen(1)
client_socket, address = s.accept()
print("Connection from: " + str(address))

#--------------------FOR TCP CLIENT---------------------
#s.connect((host,port))

def first_boot():
    label.config(text="Connectat com a:  "+ host + "\n Iniciant . . .", font=("Arial", 13))
    label.pack(expand=True)

def get_data():
    data = client_socket.recv(1024).decode()
    return str(data)

def update(name):
    if name == "":
        label.config(text="")
        label.pack(expand=True)

    elif name != "":

        if name == "0":
            label.config(text="Ho sento, no he pogut reconeixer-te")
            label.pack(expand=True)
        
        elif name == "exit":
            s.close()
            ventana.destroy()
        
        else:
            label.config(text=get_data() + " ha sigut reconegut/da")
            label.pack(expand=True)


    ventana.after(50, lambda: update(get_data()))

first_boot()

sleep(2)

ventana.after(50, lambda: update(get_data()))

ventana.mainloop()

