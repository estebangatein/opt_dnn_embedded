import torch.nn as nn
import torch

# class to build a simple tree structure
class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)

# function to get the node with a specific name
def get_node(root, name):
    if root.name == name:
        return root
    for child in root.children:
        result = get_node(child, name)
        if result:
            return result

# function to get a branch of the tree
def get_branch(node):
    if len(node.children) == 0:
        if 'softmax' in node.name:
            return [None]
        else:
            return [node.name]
        
    elif len(node.children) == 1:
        return [node.name] + get_branch(node.children[0])
    else:
        return [node.name] + node.children

# _____________________ MAIN FUNCTION _____________________
def seq_model(model):
    
    # transforms a fully functional model to a model formed by sequential models

    def visualize_model(model):
        # traces the execution of the model
        tracer = torch.fx.Tracer()
        graph = tracer.trace(model)

        # obtains a summary of the model
        graph_info = []
        for node in graph.nodes:
            node_info = {
                'name': node.name,
                'op': node.op,
                'target': str(node.target),
                'args': [str(arg) for arg in node.args],
                'kwargs': node.kwargs,
                'users': node.users
            }
            graph_info.append(node_info)

        return graph_info
    
    graph_info = visualize_model(model)

    # initializes the root node
    root = Node(graph_info[0]['name'])
    visited = [graph_info[0]['name']]

    # builds the tree structure
    for layer in graph_info[1:]:
        parent = get_node(root, layer['args'][0])
        for arg in layer['args']:
            if arg in visited:
                visited.append(layer['name'])
                parent.add_child(Node(layer['name']))

    # function that creates a partial representation of the model from the tree
    def create_model(node):
        start = get_branch(node)

        model_partial = [nn.Sequential()]
        for layer in start:
            if layer == 'x':
                continue

            if not isinstance(layer, Node) and layer is not None and 'relu' not in layer and 'flatten' not in layer:
                if isinstance(model_partial[-1], torch.nn.Flatten):
                    model_partial.append(nn.Sequential())
                model_partial[-1].add_module(layer, getattr(model, layer))

            elif not isinstance(layer, Node) and layer is not None and 'flatten' in layer:
                model_partial.append(nn.Flatten())
                
            elif not isinstance(layer, Node) and layer is not None and 'relu' in layer:
                model_partial[-1].add_module('relu', nn.ReLU())

            elif layer is None:
                return model_partial
            
            elif isinstance(layer, Node):
                model_partial.append(create_model(layer))

        return model_partial

    list_to_gen = create_model(root)

    write_blocks = {}
    write_forwards = {}

    # function that counts elements recursively
    def count_elements(lista):
        total = 0
        for elemento in lista:
            if isinstance(elemento, list): 
                total += count_elements(elemento)
            else:
                total += 1  
        return total

    num_elements = list(range(count_elements(list_to_gen)))

    # functions that will create the code structure
    def write_block(list_to_gen):
        for block in list_to_gen:
            if isinstance(block, list):
                write_block(block)
            else:
                write_blocks[num_elements.pop(0)] = block

    write_block(list_to_gen)
    num_elements = list(range(count_elements(list_to_gen)))

    def write_forward(list_to_gen, parent=None):
        for block in list_to_gen:
            if isinstance(block, list):

                list_to_gen = list_to_gen[list_to_gen.index(block):]
                for following_list in list_to_gen:
                    write_forward(following_list, parent)
            
                break


            elif isinstance(block, nn.Flatten):
                
                write_forwards[num_elements[0]] = f'flatten_{parent}'
                parent = num_elements[0]
                num_elements.pop(0)

            else:
                
                if block == list_to_gen[-1]:
                    write_forwards[num_elements[0]] = f'output_sequential_{parent}'
                    parent = num_elements[0]
                    num_elements.pop(0)
                else:
                    write_forwards[num_elements[0]] = f'sequential_{parent}'
                    parent = num_elements[0]
                    num_elements.pop(0)


    write_forward(list_to_gen)

    # function that will generate the code
    def generate_model_code(blocks_dict, forward_dict, filename="generated_model.py"):
   
        with open(filename, "w") as f:

            f.write("import torch\n")
            f.write("import torch.nn as nn\n\n")
            f.write("class GeneratedModel(nn.Module):\n")
            f.write("    def __init__(self):\n")
            f.write("        super(GeneratedModel, self).__init__()\n\n")
            
            # block definitions
            for block_id, block in blocks_dict.items():
                if not isinstance(block, nn.Flatten):  
                    if isinstance(block, nn.Sequential):
                        layers = [
                            f"nn.{str(layer)}" 
                            for layer in block
                        ]
                        block_code = f"nn.Sequential(\n            " + ",\n            ".join(layers) + "\n        )"
                    else:
                        block_code = f"nn.{str(block)}"
                    
                    f.write(f"        self.block_{block_id} = {block_code}\n\n")
            

            f.write("    def forward(self, x):\n")
            
            outputs = [] 
            for key, value in forward_dict.items():
                parts = value.split("_")
                block_type = parts[0]
                connection = parts[-1] 
                

                if connection == "None":
                    input_var = "x"
                else:
                    input_var = f"out_{connection}"
                

                if isinstance(blocks_dict[key], nn.Flatten):
                    flatten_params = blocks_dict[key]
                    f.write(f"        out_{key} = {input_var}.flatten(start_dim={flatten_params.start_dim}, end_dim={flatten_params.end_dim})\n")
                else:
                    f.write(f"        out_{key} = self.block_{key}({input_var})\n")
                
                if block_type == "output":
                    outputs.append(f"out_{key}")
            
            if outputs:
                if len(outputs) > 1:
                    outputs_list = ", ".join(outputs)
                    f.write(f"        return {outputs_list}\n")
                else:
                    f.write(f"        return {outputs[0]}\n")
            else:
                f.write("        return None\n")
        
        print(f"Model generated and saved in {filename}")
    
    generate_model_code(write_blocks, write_forwards)
    
