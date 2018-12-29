import os

for i in range(10):
    for j in os.listdir("./faces/group_photos/" + str(i)):
    	os.rename("./faces/group_photos/" + str(i)+"/"+str(j) ,"./faces/group_photos/"+str(j))
    os.rmdir("./faces/group_photos/" + str(i))