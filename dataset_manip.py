RawFile = open("raw-train.csv", "r")
OutFile = open("train.csv", "w")
RawFile.readline();
for line in RawFile.readlines():
    splitLine = line.split(',')
    splitLine.pop(0)
    date = splitLine.pop(0)
    tempDate = date.split('-')
    date = tempDate[2]
    OutFile.write(date)
    for s in splitLine:
        OutFile.write(","+s)
OutFile.close()
RawFile.close()

RawFile = open("raw-test.csv", "r")
OutFile = open("test.csv", "w")
RawFile.readline();
for line in RawFile.readlines():
    splitLine = line.split(',')
    splitLine.pop(0)
    splitLine.pop(len(splitLine)-1)
    splitLine[len(splitLine)-1] += '\n'
    date = splitLine.pop(0)
    tempDate = date.split('/')
    date = tempDate[2]
    OutFile.write(date)
    for s in splitLine:
        OutFile.write(","+s)
OutFile.close()
RawFile.close()
