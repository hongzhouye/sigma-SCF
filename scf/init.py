"""
Initializations for scf calculations:
    - preparing integrals
    - parsing & packing parameters
"""
import yaml
import psi4wrapper as p4wi

'''
scf_params = None
ao_ints = None
e_ZZ_repul = None
'''


def parse_yaml(file_name):
    """
    parses a yaml file and returns a python dict
    """
    dict_to_return = None
    with open(file_name, 'r') as yaml_file:
        dict_to_return = yaml.load(yaml_file)
    return dict_to_return


def init(input_file):
    """
    setup the parameters for thre job
    """
    global scf_params
    global ao_ints
    global e_ZZ_repul

    defaults_location = "scf_params_default.yml"
    scf_params = parse_yaml(defaults_location)
    user_input = parse_yaml(input_file)
    for key in user_input.keys():
        if key in scf_params.keys():
            scf_params[key] = user_input[key]
    ao_ints, e_ZZ_repul, n_el, n_basis = p4w.init(scf_params)
    scf_params['nel'] = n_el - scf_params['charge']
    if (scf_params['nel'] % 2 == 1):
        raise Exception("only closed shell molecules are supported!")
    else:
        scf_params['nel'] = scf_params['nel'] / 2
    scf_params['nbas'] = n_basis

if __name__ == "__main__":
    init("test.yml")
    print(scf_params)
