# Example code from jython.org
# Note for running jython if not in PATH:
# 	In Terminal, try running:
#	`java -jar [location of jython.jar in home directory] [name of python w/ java file]`
#	For example:
#	`java -jar ~/jython2.7.2/jython.jar "testing-jython.py"`


from java.lang import System # Java import

print('Running on Java version: ' + System.getProperty('java.version'))
print('Unix time from Java: ' + str(System.currentTimeMillis()))