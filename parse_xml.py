# Import necessary packages
import argparse
import re  #This module provides regular expression matching operations similar to those found in Perl.

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# The script required input and output
ap.add_argument("-i", "--input", required=True, help="path to iBug 300-W data split XML file")
ap.add_argument("-t", "--output", required=True, help="path to output data split XML file")

args = vars(ap.parse_args())

# Define the indices of our eye coordinates
LANDMARKS = set(list(range(36, 48)))

# Define regular expression and load the original XML file:
# Parsing out eye locations from the XML file:
# use a regular expression to determine if there's a "part"
# element on any given line
PART = re.compile("part name='[0-9]+'") # Extracts part elements along with their names/indexes

# Load the contents of the original XML file and open the output file for writing

print("[INFO] parsing data split XML file..")
rows = open(args["input"]).read().strip().split("\n") # Loads content of input XML file
output = open(args["output"], "w") # Opens output XML file for writing

# Loop over the input XML file to find and extract the eye landmarks:

for row in rows:
    # Loop over ther rows of the input XML file.
    # Check if the current line has (x, y)-coordinates for the facial landmarks we're interested in
    parts = re.findall(PART, row) # Find all PART in row

    # if there's no info related to the coordinates
    # related to facial landmarks, we write to disk 
    # with no further modifications
    if len(parts) == 0:
        output.write("{}\n".format(row))

    # Otherwise, there is annotation information we must process
    else: # Parse it further
        # parse out the name of the attribute from the row
        attr = "name='"
        i = row.find(attr)
        j = row.find("'", i + len(attr) + 1)
        name = int(row[i + len(attr):j])

        # if the facial landmark name exists
        # within the range of our indexes, write to our
        # output file
        if name in LANDMARKS: 
            output.write("{}\n".format(row))

# Close the output file
output.close()