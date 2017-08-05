"""
Initializations for scf calculations:
    - preparing integrals
    - parsing & packing parameters
"""
import yaml
import psi4wrapper as p4w
import os

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

    defaults_location = os.path.dirname(__file__) + "/scf_params_default.yml"
    scf_params = parse_yaml(defaults_location)
    user_input = parse_yaml(input_file)
    for key in user_input.keys():
        if key in scf_params.keys():
            scf_params[key] = user_input[key]
    ao_ints, e_ZZ_repul, n_el_tot, n_basis = p4w.init(scf_params)
    n_el_tot -= scf_params['charge']    # total num of electrons
    n_el_diff = scf_params['spin'] - 1  # n_el_alpha - n_el_beta
    if (n_el_tot + n_el_diff) % 2 == 1:
        raise Exception("Charge and spin are not compatible.")
    else:
        scf_params['nel_alpha'] = int((n_el_tot + n_el_diff) / 2)
        scf_params['nel_beta'] = int((n_el_tot - n_el_diff) / 2)
    if scf_params['spin'] is not 1:     # if not singlet
        scf_params['unrestricted'] = True
    scf_params['nbas'] = n_basis
    print(scf_params)
    return ao_ints, scf_params, e_ZZ_repul


if __name__ == "__main__":
    init("test.yml")
    print(scf_params)
