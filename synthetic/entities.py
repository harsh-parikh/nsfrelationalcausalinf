

def generate(parameters):
    '''
        Generate an entity set containing professors, students
        and courses.
    '''

    E = {}
    
    E["p"] = {i for i in range(parameters.N_p)}
    E["s"] = {i for i in range(parameters.N_s)}
    E["c"] = {i for i in range(parameters.N_c)}
    
    return E
