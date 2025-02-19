import numpy as np
import json
# import keyboard as kb



#initializing matrices
tempcords = np.zeros((1,3))
T = np.zeros((3,3))

# Assuming ref Frame as rectangular frame with location at world origin
refAxis =   [
            [1,1,1]
            ]

cam = 0

while True:
    # Accessing camera database accessing the camera location and orientatiotn

    with open("workspaces.json","r") as file:
        data = json.load(file)
        # file.close()
    camAxis = data[str(cam)]['ori']
    camLoc = data[str(cam)]['loc']
    for i in range(3):                              # converting the 1-D array to 3D numpy array
        camAxis[0][i] = np.cos(camAxis[0][i])


    # Fucntion calculating the final transformed vector 
    def calculate(cAxis,Cloc,lcoords):
        T = np.dot(np.transpose(refAxis),cAxis)
        finalVec = np.dot(T,lcoords) + Cloc
        return np.transpose(finalVec)

    # Writing the final transfromed vectore from calculate method to the End DB
    def dumper(id : str,vec,db : dict):
        db[id]["x"] = vec[0][0]
        db[id]["y"] = vec[0][1]
        db[id]["z"] = vec[0][2]

    # Accessing raw unprocessed coordinates from the camera stream
    with open("dbRaw.json","r") as Cfile:
        Cdata = json.load(Cfile)
        # Cfile.close()

    # exctracting coordinates and implementing the backend
    for locData in Cdata:
            tempcords[0][0] = Cdata[locData]["x"]
            tempcords[0][1] = Cdata[locData]["y"]
            tempcords[0][2] = Cdata[locData]["z"]
            dumper(locData,calculate(camAxis,np.transpose(camLoc),np.transpose(tempcords)),Cdata)

    with open("db.json","w") as rfile:
        json.dump(Cdata,rfile)
        # rfile.close()
