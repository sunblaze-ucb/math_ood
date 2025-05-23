from pdb import main
import pdb
import random

import numpy as np
from classical_generator import ClassicalGenerator, parse_args, main as base_main
import geo_types as gt
from typing import List, Tuple, Generator
from random_constr import Element, Command, ConstCommand
# this one is constrained to generate problems which at some point involve constructing a polygon, 
# and at one point rotating it around its center.
# later in the pipeline, the generated problem when translated into NL will have a different form:
# the "answer" to the measure will be given, and the new question will be to find the angle of rotation, which will be omitted.
class PolygonRotationGenerator(ClassicalGenerator):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.made_polygon_already: bool = False
        self.rotated_polygon_already: bool = False
        self.num_diagonal_constructions = random.randint(20, 30) # won't necessarily all end up in longest construction; in fact, the vast majority will not

    # override 
    def _sample_commands(self) -> Generator[str, None, None]:
        if not self.made_polygon_already:
            yield "polygon_from_center_and_circumradius" # note: this doesn't work,  you have to construct the args first... try yielding those above and setting true after; check resulting files

        if len(self.all_segments) < self.num_diagonal_constructions:
            yield "diagonal_p"
        command_names = list(self.available_commands.keys())
        # encourage rotation of a polygon, but only after we've done a few other things
        # rotating a polygon right after its construction is likely boring.
        if not self.rotated_polygon_already and len(self.command_sequence) > 8:
            command_names.append('rotate_polygon_about_center')
            command_names.append('rotate_polygon_about_center')
        for _ in range (3):
            command_names.append('orthogonal_line_pl')
            command_names.append('orthogonal_line_ps')
        random.shuffle(command_names)
        
        for cmd_name in command_names:
            if 'prove' in cmd_name or 'measure' in cmd_name:
                continue # these are special commands, not part of constructions
            # heuristics for not making boring things
            # also minus and power make bad (ambiguous or dependent on calculation precision) problems
            if 'minus' in cmd_name:
                continue
            if 'sum' in cmd_name:
                continue
            if 'ratio' in cmd_name:
                continue
            if 'product' in cmd_name:
                continue
            if 'power_' in cmd_name:
                continue

            # avoid constructions that are invariant under rotation
            if 'area_P' in cmd_name:
                continue
            if 'circumcircle_p' in cmd_name:
                continue

            
            cmd_info = self.available_commands[cmd_name]
            if cmd_info['return_type'] == gt.Boolean:
                continue
            # discourage multiple polygons
            if self.made_polygon_already and cmd_name == 'polygon_from_center_and_circumradius':
                if random.random() < 0.8:
                    continue
            # discourage multiple rotations
            if self.rotated_polygon_already and cmd_name == 'rotate_polygon_about_center':
                if random.random() < 0.8:
                    continue
            yield cmd_name
    
    def _try_apply_command(self, cmd_name: str, input_elements: List[Element]) -> Tuple[bool, Command]:
        result = super()._try_apply_command(cmd_name, input_elements)
        return result
    
    def compute_longest_construction(self, j):
        if not self.made_polygon_already:
            raise Exception("How did you generate something without a polygon?")
        # this can naturally already happen because we only encouraged rotations, not required by any construction
        if not self.rotated_polygon_already:
            return False
        res = super().compute_longest_construction(j)
        if not res:
            return False
        # even if we did do a rotation, it doesn't necessarily end up in the longest construction
        # many failures are okay, construction is cheap
        if not any(False if isinstance(cmd, ConstCommand) else cmd.name == 'rotate_polygon_about_center' for cmd in self.pruned_command_sequence):
            return False

        visited = set()
        relies_on_orig_polygon = False
        relies_on_rotated_polygon = False
        # now check that the measured thing in some way relies on both
        # in principle it could rely only on the rotated polygon (i.e, only rely on the original polygon indirectly through the rotated polygon), which would be lame
        def dfs(node):
            nonlocal visited
            if node.element.label in visited:
                return
            visited.add(node.element.label)
            if isinstance(node.command, ConstCommand):
                return
            if node.command.name == 'rotate_polygon_about_center':
                nonlocal relies_on_rotated_polygon
                relies_on_rotated_polygon = True
                return # don't propagate dfs from rotated polygon, because of course it relies on the original polygon
            if node.command.name == 'polygon_from_center_and_circumradius':
                nonlocal relies_on_orig_polygon
                relies_on_orig_polygon = True
                return # you can just stop here, because this is all we need to check
            for parent in node.parents:
                dfs(parent)
        
        start_node = self.dependency_graph.nodes[self.pruned_command_sequence[-1].output_elements[0]]
        # DFS from the measured node in the reverse dependency graph
        dfs(start_node)
        if not (relies_on_orig_polygon and relies_on_rotated_polygon):
            return False
        if np.count_nonzero(np.array([False if isinstance(cmd, ConstCommand) else cmd.name == 'diagonal_p' for cmd in self.pruned_command_sequence])) < 3:
            return False
        return True

if __name__ == "__main__":
    args = parse_args()
    args.generator_class = PolygonRotationGenerator
    base_main(args)