import os


folder_numbers = 10

for i in range(folder_numbers):
    for j in os.listdir("./faces/group_photos/" + str(i)):
    	os.rename("./faces/group_photos/" + str(i)+"/"+str(j) ,"./faces/group_photos/"+str(j))
    os.rmdir("./faces/group_photos/" + str(i))
