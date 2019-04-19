import numpy

class Summarization:
    """
        A feature of an entity that is the sum of other
        features.
    """
    def __init__(self, target, sieve, mapper, reducer):
        self.target = target
        self.sieve = sieve
        self.mapper = mapper
        self.reducer = reducer

    def measure(self, parent, schema) -> float:
        candidates = schema['entities'][self.target]
        confirmed = filter(self.sieve(parent, schema), candidates)
        mapped = [self.mapper(x, schema) for x in confirmed]
        result = self.reducer(mapped)
        return result


class EndogenousVar:
    def __init__(self, reducer, variables : {Summarization}):
        """
            Define a new partially endogenous variable.

            name = reducer(vars[0] + ... + vars[n])
        """
        self.reducer = reducer
        self.variables = variables

    def instantiate(self, parent, schema):
        """
            Return a concrete instance of this endogenous variable.
        """
        concrete_vars = [v.measure(parent, schema) for v in self.variables]
        return self.reducer(concrete_vars)






