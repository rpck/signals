import math
import numpy as np
import argparse
import os

def seperate(Input, setPath, bArraySize):

	readIO = open(Input, "rb")
	buffA = readIO.read(bArraySize * bArraySize)

	s = 0
	fName = setPath + "Set"
	index = 0

	while(buffA):
		newFile = fName + str(s) + ".csv"
		writeIO = open(newFile, "wb+")
		s += 1
		while (index < bArraySize):
			writeIO.write(bytes(buffA[index:index + 4]))
			writeIO.write(b",")
			index += 4
		writeIO.close()
		index = 0
		buffA = readIO.read(bArraySize * bArraySize)
	readIO.close()		


parser = argparse.ArgumentParser()
parser.add_argument('File_name', type=str)
parser.add_argument('Set_path', type=str)
parser.add_argument('Array_size', type=int)
args = parser.parse_args()
Input = args.File_name
setPath = args.Set_path
bArraySize = args.Array_size
if not os.path.exists(setPath):
    os.makedirs(setPath)

seperate(Input, setPath, bArraySize)


	


