# Base code from DialogOS project, Jython demo client
# https://github.com/dialogos-project/jython-demo-client
#
# To note: "Implementing your own Jython-based client amounts to 
# implementing your own version of client.py, which should implement 
# a class that is derived from the Client class from DialogOS. 
# See the documentation of that class. You can then create an object 
# of your class and call its open method to start the client 
# and wait for connections from DialogOS."
#

from com.clt.dialog.client import Client


class Main(Client):
    def __init__(self):
        pass

    def stateChanged(self, cs):
        print "new state: " + str(cs)

    def sessionStarted(self):
        print "session started"

    def reset(self):
        print "reset"

    def output(self, value):
        print "output: " + value.getString()

    def getName(self):
        return "Jython demo client"

    def error(self, throwable):
        print "error"


m = Main()
m.open(8888)


