import traceback
import numpy as np
np.seterr(all='raise') # RuntimeWarnings like divide by zero, degenerate determinants, etc. will now raise exceptions, invalidating some constructions.
import random
import inspect
import sys
import os
import argparse
from typing import Dict, List, Set, Tuple, Any, Union, Optional, Generator, Callable
import pdb
import concurrent.futures
# Import the commands module
import commands
import geo_types as gt
from geo_types import MEASURABLE_TYPES, AngleSize
from random_constr import Command, Element, ConstCommand
from translate_utils import invert_pi_expression
from sample_config import get_commands, triangle_commands, polygon_commands, circle_commands

class Node:
    """A node in the dependency graph representing an Element."""
    def __init__(self, element: Element, command: Optional[Command] = None):
        self.element = element
        self.command = command  # Command that generated this element
        self.parents = []  # Elements used as arguments to create this element
        self.ancestor_count = 0  # Number of ancestors (computed during dependency graph construction)
        
    def add_parent(self, parent_node: 'Node'):
        """Add a parent node (argument used to create this element)."""
        if parent_node not in self.parents:
            self.parents.append(parent_node)
            
    def __repr__(self):
        parent_labels = [p.element.label for p in self.parents]
        return f"Node({self.element.label}, parents={parent_labels}, ancestors={self.ancestor_count})"

class DependencyGraph:
    """A directed graph tracking Element dependencies in constructions."""
    def __init__(self):
        self.nodes: Dict[Element, Node] = {}

    def add_node(self, element: Element, command: Optional[Command] = None) -> Node:
        """Add a node to the graph."""
        if element not in self.nodes:
            self.nodes[element] = Node(element, command)
        return self.nodes[element]
        
    def add_dependency(self, child_element: Element, parent_elements: List[Element], command: Command):
        """
        Add a dependency relationship: child depends on parents through command.
        If nodes don't exist, they will be created.
        
        Args:
            child_element: Element for the child node
            parent_elements: List of parent elements
            command: The Command object that created the child element
        """
        # Ensure all nodes exist
        child_node = self.add_node(child_element, command)
        
        # Add parents
        for parent_element in parent_elements:
            parent_node = self.add_node(parent_element)
            child_node.add_parent(parent_node)
            
        # Update ancestor count for this node
        child_node.ancestor_count = self._calculate_ancestor_count(child_node)
    
    def _calculate_ancestor_count(self, node: Node) -> int:
        """
        Calculate the number of ancestors for a node.
        
        The count includes:
        - The sum of all ancestors of parent nodes
        - Plus one for each parent node itself
        """
        count = 0
        
        # Count direct parents
        direct_parents = len(node.parents)
        
        # Add the sum of all ancestors from parent nodes
        for parent in node.parents:
            count += parent.ancestor_count # I know this isn't actually correct; heuristic.
        
        # Add the direct parents count
        count += direct_parents
        
        return count
    
    def __repr__(self):
        return f"DependencyGraph with {len(self.nodes)} nodes"

class ClassicalGenerator:
    def __init__(self, seed=None, command_types=None):
        """Initialize the generator with a random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.init_identifier_pool()
        
        # Keep track of used identifiers and their types
        self.identifiers: Dict[str, Element] = {}
        self.identifier_queue: List[str] = []
        
        # Get all available commands from the commands module
        self.available_commands = self._get_commands()
        self.command_types = command_types
        if command_types:
            self.available_commands = {k: v for k, v in self.available_commands.items() if k in get_commands(command_types)}

        self.command_sequence: List[Command] = []
        self.dependency_graph = DependencyGraph()
        self.pruned_command_sequence: List[Command] = []
        self.made_polygon_already: bool = False
        self.made_triangle_already: bool = False
        # this has to exist because when we construct a polygon, the vertices and polygon are not naturally associated at the Element level.
        # the polygon is naturally aware of the vertices, but not necessarily of the Elements representing them, which is needed for the analysis done in this script.
        self.poly_to_vertices: Dict[Element, List[Element]] = {}

        self.all_lines: Dict[gt.Line, bool] = {} # includes segments, rays, and lines
        self.all_points: Dict[gt.Point, bool] = {}
        self.all_circles: Dict[gt.Circle, bool] = {}
        self.all_triangles: Dict[gt.Triangle, bool] = {}

    def init_identifier_pool(self):
        self.identifier_pool = [chr(i) for i in range(65, 91)]  # A-Z
        self.secondary_identifier_pool = [f"{chr(i)}{j}" for i in range(65, 91) for j in range(1, 100)]  # A1-Z99
        
    def _get_commands(self) -> Dict[str, Dict]:
        """Extract all commands from the commands module with their parameter and return types."""
        commands_dict = {}
        
        for name, func in inspect.getmembers(commands, inspect.isfunction):
            if name.startswith('_'):
                continue
                
            sig = inspect.signature(func)
            
            # Get parameter types
            param_types = []
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    param_types.append(param.annotation)
                else:
                    # If no type annotation, use Any
                    param_types.append(Any)
            
            # Get return type
            return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else Any
            
            commands_dict[name] = {
                'func': func,
                'param_types': param_types,
                'return_type': return_type
            }
        return commands_dict

    def _get_unused_identifier(self, hidden: bool = False, return_sequential: int = 0) -> str:
        """Get an unused identifier from the pool."""
        # arg hidden: use the secondary pool, because this ident is going to be missing from the translation anyway and we need to conserve good idents
        id_pool = self.identifier_pool if self.identifier_pool and len(self.identifier_pool) > return_sequential and not hidden else self.secondary_identifier_pool
        if return_sequential > 0 and len(id_pool) > return_sequential:
            identifier = id_pool.pop(0) # the next time we call this function, we will, by construction, get the next index
        else:
            idx = random.randrange(len(id_pool))
            identifier = id_pool.pop(idx)
        return identifier

    # Identifiers assigned in this specific way are not automatically removed from the identifier pool,
    # so user needs to make sure to do so themselves.
    def _assign_specific_identifiers(self) -> str:
        if self.identifier_queue:
            return self.identifier_queue.pop(0)
        else:
            print("This wasn't supposed to happen. Ran out of identifiers to assign intentionally; user should have specified a longer queue before trying to call this function.")
            return self._get_unused_identifier(return_sequential=0)

    def _is_compatible_type(self, value_type, required_type) -> bool:
        return value_type == required_type or required_type == Any

    def _find_compatible_elements(self, required_type, angle_biases: Optional[List[float]] = None) -> List[Element]:
        """Find elements that have compatible types with the required type."""
        compatible = []
        
        # Check if we need a numeric type
        numeric_type = None
        if required_type == int or required_type == float:
            numeric_type = required_type
        elif hasattr(required_type, "__origin__") and required_type.__origin__ is Union:
            # Check if any of the union types are numeric
            for t in required_type.__args__:
                if t == int or t == float:
                    numeric_type = t
                    break
                
        # First try to find existing compatible elements
        for element in self.identifiers.values():
            actual_type = type(element.data)
            if self._is_compatible_type(actual_type, required_type):
                compatible.append(element)

        if required_type == gt.AngleSize or numeric_type:
            if len(self.command_sequence) > 50:
                return [] # too many consts near the end of sequences, leading to degen problems where the end of the sequence is the only important part.
        if required_type == gt.AngleSize:
            # construct a new AngleSize
            if angle_biases is None:
                angle = random.choice(np.pi / 12 * np.arange(1, 13))
            else:
                angle = random.choice(angle_biases)
            element, const_command = self._add_constant('AngleSize', angle)
            return [element]
                
        # If this is a numeric type (int or float)
        if numeric_type is not None:            
            # Create a new constant with a random value
            if compatible and random.random() < 0.2: # make a new const with the same value; maybe it'll be interesting
                value = random.choice(compatible).data
            else:
                value = random.randint(1, 12)
            element, const_command = self._add_constant('int', value)
            # Return the newly created element
            return [element]
        
        return compatible
    
    def _update_dependency_graph(self, command: Command) -> None:
        if not self.dependency_graph:
            return
        input_elements = command.input_elements
        output_elements = command.output_elements
        # Add dependencies to the graph
        for result_elem in output_elements:
            # Add node and dependencies
            self.dependency_graph.add_node(result_elem, command)
            self.dependency_graph.add_dependency(result_elem, input_elements, command)

    def _add_constant(self, const_type: str, value: Any) -> Tuple[Element, ConstCommand]:
        identifier = self._get_unused_identifier()
        
        # Create the element
        element = Element(identifier, self.identifiers)
        
        if const_type == 'int':
            data = int(value)
        elif const_type == 'float':
            data = float(value)
        elif const_type == 'AngleSize':
            data = AngleSize(value)
        const_command = ConstCommand(type(data), value, element)        
        
        const_command.apply()
        self.command_sequence.append(const_command)
        
        # Add to dependency graph
        if self.dependency_graph:
            self.dependency_graph.add_node(element, const_command)
            self.dependency_graph.add_dependency(element, [], const_command)
            
        return element, const_command

    def _try_apply_command(self, cmd_name: str, input_elements: List[Element], label_factory: Callable[[], str] = None) -> Tuple[bool, Command]:
        """
        Try to execute a command and return its result.
        
        Args:
            cmd_name: Name of the command to execute
            input_elements: List of Element objects containing the input data
            
        Returns:
            Tuple of (success, command):
            - success: Boolean indicating if the command executed successfully
            - command: The Command object that was executed
        """
        if label_factory is None:
            label_factory = self._get_unused_identifier
        try:
            if 'intersect' in cmd_name:
                l1_constructed_by = input_elements[0].command
                l2_constructed_by = input_elements[1].command
                if 'orthogonal' in l1_constructed_by.name or 'orthogonal' in l2_constructed_by.name or 'bisector' in l1_constructed_by.name or 'bisector' in l2_constructed_by.name or 'tangent' in l1_constructed_by.name or 'tangent' in l2_constructed_by.name:
                    # we are trying to find the intersection of two lines, but likely one of them was constructed by being put through the other...
                    return False, None
            command = Command(cmd_name, input_elements, label_factory=label_factory, label_dict=self.identifiers)
            command.apply()
            failed_command = False
            for output_elem in command.output_elements:
                try:
                    if isinstance(output_elem.data, gt.Line):
                        n = output_elem.data.n
                        c = output_elem.data.c
                        key = (tuple(n), c)
                        for other_key in self.all_lines: # don't make two overlapping lines
                            other_n, other_c = other_key    
                            if np.linalg.norm(n - other_n) < 1e-4 and np.linalg.norm(c - other_c) < 1e-4:
                                failed_command = True
                                break
                            if np.linalg.norm(-n - other_n) < 1e-4 and np.linalg.norm(-c - other_c) < 1e-4:
                                failed_command = True
                                break
                        self.all_lines[key] = True
                    if isinstance(output_elem.data, gt.Point):
                        key = tuple(output_elem.data.a)
                        for other_key in self.all_points:
                            if np.isclose(key[0], other_key[0]) and np.isclose(key[1], other_key[1]):
                                failed_command = True
                                break
                        self.all_points[key] = True
                    if isinstance(output_elem.data, gt.Circle):
                        key = (tuple(output_elem.data.c), output_elem.data.r)
                        for other_key in self.all_circles:
                            if np.isclose(key[0][0], other_key[0][0]) and np.isclose(key[0][1], other_key[0][1]) and np.isclose(key[1], other_key[1]):
                                failed_command = True
                                break
                        self.all_circles[key] = True
                    if isinstance(output_elem.data, gt.Triangle):
                        key = tuple(sorted((output_elem.data.a, output_elem.data.b, output_elem.data.c), key=lambda x: x.__repr__())) # sort by string representation to make the key invariant to the order of the points
                        if key in self.all_triangles:
                            failed_command = True
                            break
                        self.all_triangles[key] = True
                except Exception as e:
                    traceback.print_exc()
                    pdb.set_trace() # this one really isn't supposed to happen
            if failed_command:
                for output_elem in command.output_elements:
                    # undo the command, which here means we don't keep track of the elements it generated
                    if output_elem.label in self.identifiers:
                        del self.identifiers[output_elem.label]
                return False, None
            if 'rotate_polygon' in cmd_name or cmd_name == 'polygon_from_center_and_circumradius':
                self.poly_to_vertices[command.output_elements[-1]] = command.output_elements[:-1]
            if cmd_name == 'triangle_ppp':
                self.made_triangle_already = True
            return True, command
        except Exception as e:
            # traceback.print_exc()
            return False, None

    def _sample_commands(self) -> Generator[str, None, None]:
        # Shuffle commands to try
        command_names = list(self.available_commands.keys())
        if not self.made_triangle_already and "triangle_ppp" in command_names:
            command_names.append('triangle_ppp') # double the probability of triangle construction
        # only sample equilateral triangle at the beginning of the sequence, since it's pretty weird to construct 3 points by fiat with no relation to anything else, kind of like polygon...
        if len(self.command_sequence) == 0:
            if 'equilateral_triangle' in command_names and random.random() < 0.0: # disabled for now
                self.command_sequence = []
                self.dependency_graph = DependencyGraph()
                self.identifiers = {}
                yield 'equilateral_triangle'
            else:
                yield 'point_'
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
            # this one is boring, since if you get a number out of something, you can't get anything more useful out of it, and if have have a polygon, you always have its circumradius, so you always have its area.
            if cmd_name == 'area_P':
                continue

            # just don't sample more than one point, since it can lead to more degenerate problems
            if cmd_name == 'point_':
                continue
            
            cmd_info = self.available_commands[cmd_name]
            if cmd_info['return_type'] == gt.Boolean:
                continue

            if cmd_name == 'equilateral_triangle' and len(self.command_sequence) > 1:
                continue

            if cmd_name == 'point_pm' and len(self.command_sequence) > 10:
                continue
            # decrease frequency of polygons
            if cmd_name == 'polygon_from_center_and_circumradius':
                # make sure they only happen near the beginning of the sequence, which is more natural
                if len(self.command_sequence) > 3:
                    continue
                # eliminate duplicates
                if self.made_polygon_already:
                    continue
                # still there are too many
                if random.random() < 0.5:
                    continue
        
            yield cmd_name

    def _sample_polygon_sides(self) -> int:
        sides = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        weights = [1, 2, 4, 1, 4, 1, 2, 1, 4]
        return random.choices(sides, weights)[0]

    def _execute_new_command(self) -> Command:
        """
        Sample a random command that can be executed with existing elements.
        Returns a tuple of (input_elements, command) or None if no valid command can be sampled.
        """
        for cmd_name in self._sample_commands():
            cmd_info = self.available_commands[cmd_name]
            param_types = cmd_info['param_types']

            # Try to find compatible parameters for the sampled command, ensuring they are unique
            valid_params = True
            if cmd_name == 'diagonal_p':
                # special semantics for this command, kind of a hack
                # simply sample two points and construct a segment, 
                # but these two points have to be vertices of a polygon.
                compatible = self._find_compatible_elements(gt.Polygon)
                if not compatible:
                    continue
                polygon_element: Element = random.choice(compatible)
                n = len(polygon_element.data.points)
                while True:
                    i, j = random.sample(range(n), 2)
                    if abs(i - j) % n != 1 and abs(i - j) % n != n - 1:
                        break
                # no need to construct the points, since they were already constructed
                # as vertices of the polygon
                points = self.poly_to_vertices[polygon_element]
                input_elements = [points[i], points[j]]
            else:
                input_elements = []
                used_elements = set()  # Track used elements to ensure uniqueness
                
                for param_type in param_types:
                    angle_biases = None
                    if cmd_name == 'rotate_polygon_about_center' and param_type == gt.AngleSize:
                        num_sides = len(input_elements[0].data.points)
                        angle_biases = [np.pi / num_sides * i for i in range(1, num_sides)] # rotate by multiples of one-half internal angle
                    compatible = self._find_compatible_elements(param_type, angle_biases)
                    if not compatible:
                        valid_params = False
                        break
                    
                    # Filter out already used elements
                    available_elements = [elem for elem in compatible if elem not in used_elements]
                    if not available_elements:
                        valid_params = False
                        break
                    
                    # Select a random compatible element
                    selected = random.choice(available_elements)
                    input_elements.append(selected)
                    used_elements.add(selected)
            if valid_params:
                if cmd_name == 'polygon_from_center_and_circumradius':
                    num_sides = self._sample_polygon_sides()
                    element, _ = self._add_constant('int', num_sides)
                    input_elements[0] = element
                # Try to execute the command
                success, command = self._try_apply_command(cmd_name, input_elements)
                if success:
                    return command
                else:
                    continue
        # this actually can happen now, since we can get unlucky and when we sample a valid command, sample the wrong args for the command, thereby skipping it.
        return None


    def generate_construction(self, num_commands: int = 5) -> List[Command]:
        """Generate a sequence of commands to form a valid construction."""
        while len(self.command_sequence) < num_commands:
            # Sample a command that can be executed
            command = self._execute_new_command()
            if command is None: # should not actually happen
                continue
            self.command_sequence.append(command)
            self._update_dependency_graph(command)
        
        return self.command_sequence

    def prune_construction(self, min_num_commands: int = 8):
        """
        Prune the construction to include only the commands needed to construct the longest quantity. Do this to make it faster to produce longer constructions.
        """
        target_node = max(self.dependency_graph.nodes.values(), key=lambda node: node.ancestor_count)
        required_commands = self.find_necessary_commands(target_node)

        self.pruned_command_sequence = required_commands

        #reset the dependency graph
        self.dependency_graph = DependencyGraph()
        for command in self.pruned_command_sequence:
            self._update_dependency_graph(command)
        
        self.identifiers = {}
        for command in self.pruned_command_sequence:
            for output_elem in command.output_elements:
                self.identifiers[output_elem.label] = output_elem.label

    def find_necessary_commands(self, target_node: Node):
        """
        Find the commands needed to construct the target node.
        """
        required_commands = set()
        visited = set()

        def dfs(node):
            if node.element.label in visited:
                return
            visited.add(node.element.label)
            for parent in node.parents:
                dfs(parent)
            if node.command:
                required_commands.add(node.command)
        
        dfs(target_node)
        # Convert to list and sort by original command order
        command_order = {cmd: i for i, cmd in enumerate(self.command_sequence)}
        ordered_commands = sorted(required_commands, key=lambda cmd: command_order.get(cmd, float('inf')))
        return ordered_commands

    def compute_longest_construction(self, j, min_num_commands: int = 8):
        """
        Find the measurable quantity with the most ancestors and create a pruned
        construction sequence that includes only the commands needed to construct it.
        """
        # Find all measurable quantities
        measurable_nodes = []
        for label, node in self.dependency_graph.nodes.items():
            element = node.element
            # Check if the element's data is a measurable type
            if any(isinstance(element.data, m_type) for m_type in MEASURABLE_TYPES):
                measurable_nodes.append(node)
        while True:
            if not measurable_nodes:
                # If no measurable quantities found, keep the original sequence
                self.pruned_command_sequence = self.command_sequence.copy()
                return False
            # Find the measurable quantity with the most ancestors
            target_node = max(measurable_nodes, key=lambda node: node.ancestor_count)
            if target_node.command.name == 'chord_c' or target_node.command.input_elements[0].command.name == 'chord_c':
                # degenerate construction, chord is constructed by length, and we are either measuring it directly or one of the points that came out of it
                measurable_nodes.remove(target_node)
                continue
            if target_node.command.name == 'radius_c':
                radius_command = target_node.command
                radius_found_by = radius_command.input_elements[0].command.name
                if radius_found_by in ('circle_pp', 'circle_pm', 'mirror_cp', 'mirror_cl') or 'tangent' in radius_found_by:
                    # degenerate construction, we started with the radius, contructed a circle, and then measured the radius.
                    measurable_nodes.remove(target_node)
                    continue
            if target_node.command.name == 'distance_pp' or target_node.command.name == 'segment_pp':
                inputs = target_node.command.input_elements
                for input_order in ((inputs[0], inputs[1]), (inputs[1], inputs[0])):
                    p2_constructed_by = input_order[1].command
                    if (p2_constructed_by.name == 'mirror_pp' and p2_constructed_by.input_elements[1] == input_order[0]) \
                    or ('rotate' in p2_constructed_by.name and p2_constructed_by.input_elements[2] == input_order[0]):
                        # degenerate construction, we measured a mirrored/rotated point's distance, which is the same distance.
                        measurable_nodes.remove(target_node)
                    continue
                dist_names = ('point_pm', 'point_at_distance_along_line', 'point_c', 'translate_pv')
                if target_node.command.input_elements[0].command.name in dist_names or target_node.command.input_elements[1].command.name in dist_names:
                    # degenerate construction, we started with the distance, constructed a point, and then measured the distance.
                    # strictly speaking we need to check that the other arg is the other point in distance_pp, but i don't care, we can throw away some extras.
                    measurable_nodes.remove(target_node)
                    continue
            if target_node.command.name == 'angle_ppp':
                if not 'pi' in invert_pi_expression(target_node.element.data.angle):
                    measurable_nodes.remove(target_node)
                    continue
                if np.isclose(target_node.element.data.angle, np.pi): # degenerate angle, things lie on a straight line...
                    measurable_nodes.remove(target_node)
                    continue
                if np.isclose(target_node.element.data.angle, np.pi/2): # probably constructed explicitly via perpendicular line, so kind of stupid
                    measurable_nodes.remove(target_node)
                    continue
                p3_constructed_by = target_node.command.input_elements[2].command
                p1_constructed_by = target_node.command.input_elements[0].command
                if 'rotate' in p1_constructed_by.name or 'rotate' in p3_constructed_by.name:
                    # this condition isn't strict enough, but basically paranoidly remove dumb angle constructions
                    measurable_nodes.remove(target_node)
                    continue
            break
        
        ordered_commands = self.find_necessary_commands(target_node)

        # Add a measure command for the target node
        # Create a new measure command for the target element
        measure_command = Command('measure', [target_node.element], label_dict=self.identifiers)
        measure_command.apply()
        self._update_dependency_graph(measure_command)
        ordered_commands.append(measure_command)        


        '''
        # if most recent few commands had a literal arg, it probably caused the problem to be dumb because the end of the sequence is the only important part.
        for cmd in ordered_commands[-7:]:
            if isinstance(cmd, ConstCommand):
                return False
            if cmd.name == 'translate_pv':
                return False
        '''

        # reassign identifiers starting from the beginning of the ident pool (shuffled, but with single char idents first),
        # so that the output is more readable
        # NOTE: this means all ident logic applied before here is useless
        # a funny thing happened here:
        # you don't have to rename the input elements, because the semantics of the rest of this program involve manipulating actual element objects.
        # so when you rename the output elements, you will rename every element which is actually used in the command sequence,
        # and the element object will be updated, passed around, and have the correct label when it is used as an input element.
        # in fact, trying to rename the input element will fail, because it has already been renamed.
        self.init_identifier_pool()
        num_nonconst_commands = 0
        for command in ordered_commands:
            if isinstance(command, ConstCommand):
                command.element.label = self._get_unused_identifier(hidden=True)
                continue
            num_nonconst_commands += 1
            if command.name == 'polygon_from_center_and_circumradius':
                num_sides = command.input_elements[0].data
                for i in range(num_sides):
                    output_elem = command.output_elements[i]
                    output_elem.label = self._get_unused_identifier(return_sequential=num_sides - i)
                command.output_elements[-1].label = self._get_unused_identifier() # the polygon itself
            elif command.name == 'rotate_polygon_about_center':
                for idx, output_elem in enumerate(command.output_elements[:-1]):
                    input_vertex = self.poly_to_vertices[command.input_elements[0]][idx]
                    output_elem.label = input_vertex.label + "'"
                command.output_elements[-1].label = command.input_elements[0].label + "'"
            else:
                ident_will_be_hidden = False
                if command.name in ('segment_pp', 'angle_ppp', 'triangle_ppp', 'diagonal_p'):
                    ident_will_be_hidden = True
                for output_elem in command.output_elements:
                    output_elem.label = self._get_unused_identifier(hidden=ident_will_be_hidden)
                if command.name == 'equilateral_triangle':
                    command.output_elements[0].label = self._get_unused_identifier(hidden=True)
                    command.output_elements[1].label = self._get_unused_identifier(hidden=False)
                    command.output_elements[2].label = self._get_unused_identifier(hidden=False)
                    command.output_elements[3].label = self._get_unused_identifier(hidden=False)
                    command.output_elements[4].label = self._get_unused_identifier(hidden=True)
                    command.output_elements[5].label = self._get_unused_identifier(hidden=True)
                    command.output_elements[6].label = self._get_unused_identifier(hidden=True)
        if num_nonconst_commands < min_num_commands: # often so short as to be degenerate or uninteresting
            return False

        # we still need to check for this, because in principle we could always just construct something with only basic commands.
        if 'all' not in self.command_types and self.command_types != ['basic']:
            required_interesting_commands = []
            if 'triangle' in self.command_types:
                required_interesting_commands.extend(triangle_commands)
            if 'polygon' in self.command_types:
                required_interesting_commands.extend(polygon_commands)
            if 'circle' in self.command_types:
                required_interesting_commands.extend(circle_commands)
            found = False
            for cmd in ordered_commands:
                if isinstance(cmd, ConstCommand):
                    continue
                if cmd.name in required_interesting_commands:
                    found = True
                    break
            if not found:
                return False
        self.pruned_command_sequence = ordered_commands
        return True

    def save_construction(self, filename: str, description: str = "Generated construction"):
        with open(filename, 'w') as f:
            f.write(f"# {description}\n")
            for cmd in self.pruned_command_sequence:
                f.write(f"{cmd}\n")


def write_construction(i, args):
    seed = args.seed + i if args.seed is not None else None
    generator_class = args.generator_class
    generator = generator_class(seed=seed, command_types=args.command_types)
    generator.generate_construction(num_commands=args.num_commands)
    
    # Prune the construction to include only essential commands
    success = generator.compute_longest_construction(i, min_num_commands=args.min_num_commands)
    if not success:
        return
    
    # Create unique filename if generating multiple constructions
    filename = os.path.join(args.output_dir, f"construction_{i+1}.txt")
        
    generator.save_construction(filename, f"Generated construction #{i+1}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate classical geometric constructions")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_commands", type=int, default=25, help="Number of commands to generate")
    parser.add_argument("--output_dir", type=str, default="generated_constructions/", help="Output directory")
    # note as a result of the multiprocessing, this is the number of construction attempts, not the number of constructions actually generated
    parser.add_argument("--count", type=int, default=20, help="Number of constructions to attempt")
    parser.add_argument("--min_num_commands", type=int, default=8, help="Minimum number of commands in the output sequence")
    parser.add_argument("--max_workers", type=int, default=16, help="Maximum number of threads to use")
    parser.add_argument("--multiprocess", action="store_true", help="use multiprocessing")
    parser.add_argument("--command_types", type=str, nargs="+", choices=["polygon", "circle", "triangle", "basic", "all"], default=["all"],
                        help="Types of geometric commands to include")
    args = parser.parse_args()
    return args

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.multiprocess:
        for i in range(args.count):
            write_construction(i, args)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(write_construction, i, args) for i in range(args.count)]
            concurrent.futures.wait(futures)


if __name__ == "__main__":
    args = parse_args()
    args.generator_class = ClassicalGenerator
    main(args)
