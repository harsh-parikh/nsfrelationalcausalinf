#!/usr/bin/env python3

# built-ins
import argparse
import random
import json
import pprint

# 3rd-party libraries
import names

# local modules
import synthetic.util as util
from synthetic.structural_eqs import tutoring

parser = argparse.ArgumentParser(description='Generate schema as specified ' +
                                 'in section 7 of ...',
                                 formatter_class=argparse.
                                    ArgumentDefaultsHelpFormatter)
parser.add_argument('--save-json',
                     help='Write output to JSON file')
parser.add_argument('-p', '--professors', default=3, type=int,
                     help='Number of professors to generate')
parser.add_argument('-s', '--students', default=3, type=int,
                     help='Number of students to generate')
parser.add_argument('-c', '--courses', default=3, type=int,
                     help='Number of courses to generate')
parser.add_argument('-f', '--friendliness', default=0.25, type=float,
                     choices=[util.Range(0.0, 1.0)],
                     help='Probablity any two students are friends')
parser.add_argument('-m', '--commitment', default=0.25, type=float,
                     choices=[util.Range(0.0, 1.0)],
                     help='Probablity a student registers for any one course')
parser.add_argument('-t', '--teaching-interest', default=0.25, type=float,
                     choices=[util.Range(0.0, 1.0)],
                     help='Probablity a professor teaches any one course')
parser.add_argument('-i', '--intelligence', default=0.5, type=float,
                     choices=[util.Range(0.0, 1.0)],
                     help='Probablity a student has high intelligence')
parser.add_argument('-a', '--teaching-ability', default=0.5, type=float,
                     choices=[util.Range(0.0, 1.0)],
                     help='Probablity a professor has high teaching skill')
parser.add_argument('-d', '--difficulty', default=0.5, type=float,
                     choices=[util.Range(0.0, 1.0)],
                     help='Probablity a course has high difficulty')


'''
    Generate professor, student and course entities.
    Corresponds to \bf E in example 3.1.
'''
def generate_entities(professors, students, courses):
    p = ["Dr. " + names.get_first_name() for _ in range(professors)]
    s = [names.get_first_name() for _ in range(students)]
    c = [f'CSE {random.randint(10, 59) * 10}' for _ in range(courses)]
    return {'professors': p, 'students': s, 'courses': c}


'''
    Generate friendship, registered and teaches relations. 
    Corresponds to \bf R in example 3.1
    
    Friendships are generated using a Erdös-Renyi Model, where any
    two students have probabilty `social_skills`.
    
    Registered and teaches are generated using a random bipartite graph,
    with probabilities `commitment` and `teaching_interest` respectively.
'''
def generate_relations(entities, friendliness, commitment, interest):
    f = util.bipartite_graph(entities['students'], entities['students'],
                             friendliness, self_loops=False)
    r = util.bipartite_graph(entities['students'], entities['courses'],
                             commitment)
    t = util.bipartite_graph(entities['professors'], entities['courses'],
                             interest)
    return {'friendship': f, 'registered': r, 'teaches': t}
    

def generate_attributes(entities, intelligence, ability, difficulty):
    intelligence = {(s, 'high' if random.random() < intelligence else 'low')
                    for s in entities['students']}
    teaching_skills = {(p, 'high' if random.random() < ability else 'low') for
                        p in entities['professors']}
    difficulty = {(p, 'high' if random.random() < difficulty else 'low') for
                        p in entities['courses']}
    return {'intelligence': intelligence,
            'teaching_skills': teaching_skills,
            'difficulty': difficulty}


def generate_endogenous(schema):
    t = set()
    for c in schema['entities']['courses']:
        t.add((c, tutoring.instantiate(c, schema)))

    return {"tutoring": t}


def main():
    args = parser.parse_args()
    entities = generate_entities(args.professors, args.students, args.courses)
    relations = generate_relations(entities, args.friendliness,
                                     args.commitment, args.teaching_interest)
    attributes_exogenous = generate_attributes(entities, args.intelligence,
                                     args.teaching_ability, args.difficulty)

    schema = {"entities": entities,
              "relations": relations,
              "attributes": attributes_exogenous}

    attributes_endogenous = generate_endogenous(schema)
    schema["attributes"] = {**attributes_exogenous, **attributes_endogenous}

    if (args.save_json):
        with open(args.save_json, 'w') as f:
            json.dump(schema, f, ensure_ascii=False,
                      cls=util.SetEncoder, indent=4)
            print(f"Schema written to {args.save_json}")
    else:
        pprint.PrettyPrinter(indent=4).pprint(schema)


# execute only if run as a script
if __name__ == "__main__":
    main()
