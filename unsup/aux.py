import os
from tempfile import NamedTemporaryFile
from subprocess import run
from . import dir_dictionary


def run_all_scripts(script_dict, slurm=True):
    """this is another function that those _sub files should call. this actually execute files"""
    if slurm:
        trash_global = os.path.join(dir_dictionary['root'], 'trash')
        os.chdir(trash_global)

    for script_name, script_content in script_dict.items():
        # make sure it will run.
        assert script_content.startswith('#!/usr/bin/env bash\n')
        file_temp = NamedTemporaryFile(delete=False)
        file_temp.write(script_content.encode('utf-8'))
        file_temp.close()
        print(script_name, 'start')
        # print(script_content)
        # input('haha')
        if not slurm:
            os.chmod(file_temp.name, 0o755)
            # then run it.
            run(file_temp.name, check=True)
        else:
            run(['sbatch', file_temp.name], check=True)
        # this is fine, because sbatch actually caches the file result
        os.remove(file_temp.name)
        print(script_name, 'done')
