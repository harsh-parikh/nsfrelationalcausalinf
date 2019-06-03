'''
    This module generates a synthetic dataset, used to test the
    HUME causal inference system.
'''

from synthetic import parameters
from synthetic import entities
from synthetic import relations
from synthetic import attributes

output_file = "output.json"


def main():
    '''
        Generate a university instance, writing it to output_file
    '''
    universe = generate(parameters)
    write_json(universe, output_file)

    
def generate(parameters):
    print("generating entities")
    E = entities.generate(parameters)
    print("generating relations")
    R = relations.generate(E, parameters)
    print("generating attributes")
    A = attributes.generate(E, R, parameters)
    
    return (E, R, A)

    
def write_json(universe, file):
    pass
