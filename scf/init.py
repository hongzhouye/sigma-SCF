"""
Initializations for scf calculations:
    - preparing integrals
    - parsing & packing parameters
"""
import yaml
import psi4wrapper as p4w

scf_params = None
ao_ints = None


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
    defualts_location = "scf_params_default.yml"
    scf_params = parse_yaml(defualts_location)
    user_input = parse_yaml(input_file)
    for key in user_input.keys():
        if key in scf_params.keys():
            scf_params[key] = user_input[key]
#    ao_ints = p4w.init(scf_params)

if __name__ == "__main__":
    init("test.yml")
    print(scf_params)
