import os
import glob

# Get the list of Python module files in the current directory
module_files = glob.glob(os.path.join(os.getcwd(), '*.pyd'))

# Print the module names without the file extension
for file_path in module_files:
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    print(module_name)