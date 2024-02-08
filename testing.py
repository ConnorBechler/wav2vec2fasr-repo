#https://stackoverflow.com/a/50194143
#Solution did not work because pip install -e "D:/Northern Prinmi Data/wav2vec2faasr" somehow installs a link
# to the github package rather than the local package
#Can't find an answer online, ASK JOE

#The below line therefore fails (if testing.py is in the tests directory) as the module is not recognized
from importlib.resources import path as import_path
import pathlib
from src import forcedalignment
import tests.test_files as testingfiles

with import_path(testingfiles, "td21-22_020.wav") as test_path:
    test_path = str(test_path)
test_dir = pathlib.Path(test_path).parent
test_rec = pathlib.Path(test_path)

print(test_dir, test_rec)

