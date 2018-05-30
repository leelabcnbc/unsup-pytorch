from sys import version_info
import os

assert version_info >= (3, 6), "Python 3.6 or higher required"

root_dir = os.path.normpath(os.path.join(os.path.split(__file__)[0], '..'))
assert os.path.isabs(root_dir)

dir_dictionary = {
    'root': root_dir,
    'package': os.path.join(root_dir, 'unsup'),
    'debug': os.path.join(root_dir, 'debug'),
    'debug_reference': os.path.join(root_dir, 'debug', 'reference'),
}
