from collections import OrderedDict
from itertools import product
from unsup import aux, dir_dictionary

header = f"""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --gres=gpu:1

. activate tf15
# wait for a while. otherwise, it may not work... maybe some bug of conda.
sleep 2
cd {dir_dictionary['root']}
. ./setup_env_variables.sh

""".strip()

template_function_inner = ("PYTHONUNBUFFERED=1 python script/2010_paper/training_script.py "
                           f"{{lam_str}} {{lr_str}} {{bs_str}} "
                           f"2>&1 | tee {dir_dictionary['root']}/trash/"
                           f"sc_2010_{{lam_str}}_{{lr_str}}_{{bs_str}}"
                           ).strip()

if __name__ == '__main__':
    # prepare scripts.
    lam_list = (
        '0.25', '0.5', '0.75', '1.0', '1.25', '1.5'
    )

    lr_list = (
        '0.005', '0.001', '0.0005', '0.0001'
    )

    bs_list = (
        '4', '8', '16', '32',
    )

    script_dict = OrderedDict()

    for lam, lr, bs in product(lam_list, lr_list, bs_list):
        script_dict[f'{lam}_{lr}_{bs}'] = header + '\n\n' + template_function_inner.format(
            lam_str=lam, lr_str=lr, bs_str=bs,
        )

    aux.run_all_scripts(script_dict, slurm=True)
