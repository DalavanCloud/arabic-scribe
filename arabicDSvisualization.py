import xml.etree.ElementTree as ET
import os
filelist = []
for dirName, subdirList, fileList in os.walk("./arabicData"):
       for fname in fileList:
           if(fname.__contains__("inkml")):
              filelist.append(dirName + "/" + fname)
fname = next((fname for fname in filelist if "1231874526312.inkml" in fname), None)
fname = filelist[1]
tree = ET.parse(fname)
X = []
Y = []
durations = []
counter =0
root = tree.getroot()
dd = 2000
for child in root:
    if(counter==dd):
        break
    if (child.tag == "{http://www.w3.org/2003/InkML}trace" ):
        durationP=float(child.attrib.get("duration"))
        points = child.text.split(",")
        durations.append(durationP / len(points)/100.0)
        for point in (points):
             if( counter == dd):
                 break
             x,y = point.split(" ")
             X.append(x)
             Y.append(y)
             counter+=1



        X.append("eos")
        Y.append('eos')


print X


from xml.etree.ElementTree import Element, SubElement, tostring

rootname = "root"
root = Element(rootname)
infosChild = SubElement(root, "infos")
widthChild = SubElement(infosChild, "width")
widthChild.text = "800"
heightChild = SubElement(infosChild, "height")
heightChild.text = "600"

animationChild = SubElement(root, "animation")

startOfStrokeFlag = True
k=0
time = 0
for i in range(0, len(X)):
    if (X[i] == "eos"):
        startOfStrokeFlag = True
        k+=1
        continue
    actionChild = SubElement(animationChild, "action")
    time += durations[k]
    actionChild.set('time',str(time))
    if(startOfStrokeFlag == 1):
        startpointChild = SubElement(actionChild, "startpoint")
        startpointChild.set('x', X[i])
        startpointChild.set('y', Y[i])
        startpointChild.set('width', "3")
        startpointChild.set('color', "255")
        startpointChild.set('alpha', "0")
        startOfStrokeFlag = False
        continue

    pointChild = SubElement(actionChild, "point")
    pointChild.set('x', X[i])
    pointChild.set('y', Y[i])


tree = ET.ElementTree(root)
tree.write("test.xml")



