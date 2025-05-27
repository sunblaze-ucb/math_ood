import os, pdb
from typing import Optional
from geo_types import *
import math

type_to_shortcut = {
    int       : 'i',
    float     : 'f',
    Boolean   : 'b',
    Measure   : 'm',
    Point     : 'p',
    Polygon   : 'P',
    Circle    : 'c',
    Arc       : 'C',
    Line      : 'l',
    Ray       : 'r',
    Segment   : 's',
    Angle     : 'a',
    AngleSize : 'A',
    Vector    : 'v',
    Triangle  : 't',
}

def command_types_name(name, params):
    return "{}_{}".format(name, ''.join(type_to_shortcut[type(x)] for x in params))

import commands as commands_module
from inspect import getmembers, isfunction
command_dict = dict(o for o in getmembers(commands_module) if isfunction(o[1]))

class Element:
    def __init__(self, label, element_dict):
        if isinstance(label, dict):
            raise ValueError("Label is a dict, you meant to pass that as element_dict")
        self.data = None
        self.label = label
        self.command = None 
        if element_dict is None:
            pdb.set_trace()
        assert(label not in element_dict)
        element_dict[label] = self

    def drawable(self):
        return isinstance(self.data, (Point, Line, Angle, Polygon, Circle, Vector))
    def draw(self, cr, corners):
        if self.drawable(): self.data.draw(cr, corners)
    def important_points(self):
        if self.drawable(): return self.data.important_points()
        else: return []

    def has_value(self):
        """Check if this element has a measurable value."""
        return isinstance(self.data, MEASURABLE_TYPES)
        
    def value(self):
        """Extract the numeric value from this element."""
        if isinstance(self.data, (Measure, AngleSize)): 
            return self.data.x
        elif isinstance(self.data, Boolean): 
            return float(self.data.b)
        elif isinstance(self.data, Angle): 
            return self.data.angle
        elif isinstance(self.data, Segment): 
            return self.data.length
        elif isinstance(self.data, Polygon): 
            return commands_module.area_P(self.data).x
        elif isinstance(self.data, Triangle): 
            return commands_module.area_P(self.data).x
        elif isinstance(self.data, Circle):
            return self.data.r  # Measure circle radius
        elif isinstance(self.data, (float, int)):
            return float(self.data)
        else: 
            return None

class Command:
    def __init__(self, command_name, input_elements, output_elements=None, label_factory=None, label_dict=None):
        self.name = command_name
        self.input_elements = input_elements
        self.output_elements = output_elements
        self.label_factory = label_factory
        self.label_dict = label_dict

    def apply(self):
        # print(self)
        input_data = [x.data for x in self.input_elements]
        name = command_types_name(self.name, input_data)
        if name not in command_dict: name = self.name
        f = command_dict[name]
        output_data = f(*input_data)
        if not isinstance(output_data, (tuple, list)):
            output_data = (output_data,)
        if self.output_elements:
            if len(output_data) != len(self.output_elements):
                pdb.set_trace()
            assert(len(output_data) == len(self.output_elements))
            for x,o in zip(output_data, self.output_elements):
                if o is not None:
                    o.data = x
        else:
            self.output_elements = []
            for datum in output_data:
                label = self.label_factory() if self.label_factory else None
                self.output_elements.append(Element(label, self.label_dict))
                self.output_elements[-1].data = datum
                self.output_elements[-1].command = self

    def __repr__(self):
        inputs_str = ' '.join([x.label for x in self.input_elements])
        outputs_str = ' '.join([x.label if x is not None else "_" for x in self.output_elements])
        return "{} : {} -> {}".format(
            self.name, inputs_str, outputs_str
        )

const_type_to_str = {
    int : "int",
    float : "float",
    AngleSize : "AngleSize",
    Measure : "Measure",
}
str_to_const_type = dict((s,t) for (t,s) in const_type_to_str.items())

class ConstCommand:
    def __init__(self, datatype, value, element):
        self.datatype = datatype
        self.value = value
        self.element = element

    def apply(self):
        self.element.data = self.datatype(self.value)

    def __repr__(self):
        datatype_str = const_type_to_str[self.datatype]
        return "const {} {} -> {}".format(datatype_str, self.value, self.element.label)

def parse_command(line, element_dict):
    # Skip empty lines and comment lines
    line = line.strip()
    if not line or line.startswith('#'):
        return None
        
    tokens = line.split()
    if tokens[0] == "const":
        assert(len(tokens) == 5)
        datatype = str_to_const_type[tokens[1]]
        value = float(tokens[2])
        assert(tokens[3] == "->")
        label = tokens[4]
        element = Element(label, element_dict=element_dict)
        command = ConstCommand(datatype, value, element)
        element.command = command
        return command
    else:
        command_name = tokens[0]
        assert(tokens[1] == ":")
        labels = [None if token == "_" else token for token in tokens[2:]]
        # Filter out any None values that might have been created from extra spaces
        labels = [label for label in labels if label is not None]
        arrow_index = labels.index("->")
        input_labels = labels[:arrow_index]
        output_labels = labels[arrow_index+1:]
        input_elements = [element_dict[label] for label in input_labels]
        def element_or_none(label):
            if label is None: return None
            else: return Element(label, element_dict=element_dict)
        output_elements = list(map(element_or_none, output_labels))
        command = Command(command_name, input_elements, output_elements)
        for el in output_elements:
            if el is not None: el.command = command
        return command

class Construction:
    def __init__(self, display_size = (100,100), min_border = 0.1, max_border = 0.25):
        self.corners = np.array(((0,0), display_size))
        self.min_border = min_border
        self.max_border = max_border
        self.nc_commands = []
        self.to_prove: Optional[Element] = None
        self.to_measure: Optional[Element] = None
        self.statement_type = None  # "prove" or "measure"
        self.element_dict = dict()
        self.elements = []

    def render(self, cr, elements = None): # default: render all elements
        if elements is None: elements = self.elements

        for el in elements:
            el.draw(cr, self.corners)

    def render_to_numpy(self, elements = None):
        surface = cairo.ImageSurface(cairo.FORMAT_A8, self.width, self.height)
        cr = cairo.Context(surface)
        self.render(cr, elements)

        data = surface.get_data()
        data = np.array(data, dtype = float)/255
        data = data.reshape([self.height, surface.get_stride()])
        data = data[:,:self.width]
        return data

    def load(self, filename=None, file_contents=None):
        self.nc_commands = []
        self.to_prove = None
        self.to_measure = None
        self.statement_type = None
        self.element_dict = dict()
        self.const_commands = []
        if filename:
            with open(filename, 'r') as f:
                file_contents = f.read()
        if not file_contents:
            raise ValueError("Called Construction.load with neither filename nor file_contents")
        for line in file_contents.strip().split('\n'):
            command = parse_command(line, self.element_dict)
            if command is None:
                continue
            if isinstance(command, ConstCommand):
                self.const_commands.append(command)
                command.apply()
            elif isinstance(command, Command):
                if command.name == "prove":
                    [inp] = command.input_elements
                    [out] = command.output_elements
                    if out is not None: del self.element_dict[out.label]
                    assert(self.to_prove is None and self.to_measure is None)
                    self.to_prove = inp
                    self.statement_type = "prove"
                elif command.name == "measure":
                    [inp] = command.input_elements
                    [out] = command.output_elements
                    if out is not None: del self.element_dict[out.label]
                    assert(self.to_prove is None and self.to_measure is None)
                    self.to_measure = inp
                    self.statement_type = "measure"
                else: self.nc_commands.append(command)

        assert(self.statement_type is not None)
        assert(self.to_prove is not None or self.to_measure is not None)
        self.elements = list(self.element_dict.values())

    def run_commands(self):
        for command in self.nc_commands: command.apply()

    def generate(self, require_theorem = True, max_attempts = 100): # max_attempts = 0 -> inf
        while True:
            try:
                self.run_commands()
            except:
                max_attempts -= 1
                if max_attempts == 0: raise
                continue
            if self.statement_type == "prove" and require_theorem and not self.to_prove.data.b: 
                continue
            break

        self.fit_to_window()

    def fit_to_window(self):
        important_points = []
        for el in self.elements: important_points += el.important_points()
        if len(important_points) == 0: return
        src_corners = np.stack([
            np.min(important_points, axis = 0),
            np.max(important_points, axis = 0),
        ])
        src_size = np.maximum(0.01, src_corners[1] - src_corners[0])

        dest_size = self.corners[1] - self.corners[0]
        dest_corners_shift = np.random.random(size = [2,2])
        dest_corners_shift *= self.max_border - self.min_border
        dest_corners_shift += self.min_border
        dest_corners_shift *= np.array((1,-1)).reshape((2,1)) * dest_size
        dest_corners = self.corners + dest_corners_shift
        dest_size = dest_corners[1] - dest_corners[0]

        scale = np.min(dest_size / src_size)
        src_corners *= scale
        shift = np.average(dest_corners, axis = 0) - np.average(src_corners, axis = 0)
        for el in self.elements:
            if isinstance(el.data, (int, float)): continue
            el.data.scale(scale)
            el.data.translate(shift)

        important_points = []
        for el in self.elements: important_points += el.important_points()
        corners = np.stack([
            np.min(important_points, axis = 0),
            np.max(important_points, axis = 0),
        ])

    def test(self, num_tests = 1, verbose=True):
        if self.statement_type == "prove":
            constr_fail, check_fail, success = 0, 0, 0
            for test_idx in range(num_tests):
                try:
                    self.run_commands()
                    if self.to_prove.data.b: success += 1
                    else: check_fail += 1
                except Exception as e:
                    constr_fail += 1
                    if verbose:
                        print(f"Test {test_idx+1} failed: {str(e)}")
                        import traceback
                        traceback.print_exc()

            constr_fail, check_fail, success = [
                100*x / num_tests
                for x in (constr_fail, check_fail, success)
            ]
            print("{:.2f}% failed constructions, {:.2f}% false, {:.2f}% true".format(
                constr_fail, check_fail, success
            ))
        elif self.statement_type == "measure":
            constr_fail, measurements = 0, []
            for test_idx in range(num_tests):
                try:
                    self.run_commands()
                    measurements.append(self.to_measure.value())
                except Exception as e:
                    constr_fail += 1
                    if verbose:
                        print(f"Test {test_idx+1} failed: {str(e)}")
                        import traceback
                        traceback.print_exc()

            if constr_fail > 0:
                constr_fail_percent = 100 * constr_fail / num_tests
                print("{:.2f}% failed constructions".format(constr_fail_percent))
            
            if measurements:
                avg_measurement = sum(measurements) / len(measurements)
                print("Measurements: {} samples, avg={:.4f}, min={:.4f}, max={:.4f}".format(
                    len(measurements), avg_measurement, min(measurements), max(measurements)
                ))
        else:
            raise ValueError("Unknown statement type: {}".format(self.statement_type))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test geometric constructions')
    parser.add_argument('--test_measure', action='store_true', help='Test measure statement')
    parser.add_argument('--file', type=str, default=None, help='Specific file to test')
    args = parser.parse_args()

    construction = Construction()
    
    if args.test_measure:
        # Test measure statement
        print("Testing measure statements...")
        test_files = ["test_measure_simple.txt", "test_measure.txt", "test_measure_complex.txt", "test_measure_ratio.txt"]
        if args.file:
            test_files = [args.file]
        
        for test_file in test_files:
            print(f"\nTesting {test_file}...")
            try:
                construction.load(test_file)
                construction.test(num_tests=10)
            except Exception as e:
                print(f"Error with test file {test_file}: {str(e)}")
    else:
        # Test prove statements (original functionality)
        print("Testing prove statements...")
        datadir = "ggb-benchmark/true"
        test_files = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith(".txt")]
        
        if args.file:
            test_files = [args.file]
        
        for filename in test_files:
            print(f"\nTesting {filename}...")
            construction.load(filename)
            construction.test()
            if not args.file:  # Only test one file if not specified
                break
