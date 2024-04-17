# This is a script that will rename each image inside a folder
# with the name of the folder followed by a number increment

import os
import re

# This path needs to be adjusted
mainFolderPath = '../../Data'  # IMPORTANT: Need to make sure this is the path to the dataset folder

# Class names are the folders that are found in the data folder
classNames = ['Happy', 'Neutral', 'Engaged', 'Surprised']

for className in classNames:
    classFolderPath = os.path.join(mainFolderPath, className)

    if os.path.exists(classFolderPath):
        files = os.listdir(classFolderPath)
        i = 0
        nothingToRename = True

        for file in files:
            oldPath = os.path.join(classFolderPath, file)

            # Check if the file name is consistent
            pattern = re.compile(f"{className}_(\d+)\\.png")
            match = pattern.match(file)
            
            if match:
                # Skip files that already follow the pattern
                continue
            else:
                # Find a unique name by incrementing i until an available name is found
                nothingToRename = False # There was something to rename
                i += 1
                newName = f"{className}_{i}.png"
                while os.path.exists(os.path.join(classFolderPath, newName)):
                    i += 1
                    newName = f"{className}_{i}.png"

                newPath = os.path.join(classFolderPath, newName)
                os.rename(oldPath, newPath)

                print(f"Renamed: {file} to {newName}")
        
        if nothingToRename:
            # Prints if there were no files to rename in the specific class
            print(f'There were no files to rename in {className}')
    else:
        print(f"Class folder not found: {className}")