#
# This script renames the nodes in the .dat file
# so that they are compatible with the original inference code
#
# Original Author: Jason Guo (LBL/UCB)
# Modified for TrkQual: A. Edmonds
#
import sys

name_dict = { "tensor_sequentialdenseBiasAddReadVariableOp0 7\n" : "tensor_densebias0 7\n",
              "tensor_sequentialdense1MatMulReadVariableOp0 49\n" : "tensor_dense1kernel0 49\n",
              "tensor_sequentialdense1BiasAddReadVariableOp0 7\n" : "tensor_dense1bias0 7\n",
              "tensor_sequentialdense2BiasAddReadVariableOp0 6\n" : "tensor_dense2bias0 6\n",
              "tensor_sequentialdense2MatMulReadVariableOp0 42\n" : "tensor_dense2kernel0 42\n",
              "tensor_sequentialdense3BiasAddReadVariableOp0 1\n" : "tensor_dense3bias0 1\n",
              "tensor_sequentialdenseMatMulReadVariableOp0 49\n" : "tensor_densekernel0 49\n",
              "tensor_sequentialdense3MatMulReadVariableOp0 6\n" : "tensor_dense3kernel0 6\n"
             }

order = [
    "tensor_dense3bias0 1\n",
    "tensor_dense3kernel0 6\n",
    "tensor_dense2bias0 6\n",
    "tensor_dense2kernel0 42\n",
    "tensor_dense1bias0 7\n",
    "tensor_dense1kernel0 49\n",
    "tensor_densebias0 7\n",
    "tensor_densekernel0 49\n"
]


def sort_tensors_with_values(lines):

    tensors = {}
    #print(lines)
    for line in lines:
        if line.startswith('tensor'):
            current_label = line
            tensors[current_label] = 0
        elif current_label:
            tensors[current_label] = line
    sorted_lines = []
    for i in range(len(order)):
        sorted_lines.append(order[i])
        if i == len(order) - 1:
            sorted_lines.append(tensors[order[i]])
        else:
            sorted_lines.append(tensors[order[i]])
    #print(sorted_lines)
    with open(sys.argv[2], 'w') as file:
        file.writelines(sorted_lines)
    return sorted_lines

def rename_nodes(lines):
    renamed_lines = []
    for line in lines:
        if line in name_dict.keys():
            renamed_lines.append(name_dict[line])
        else:
            renamed_lines.append(line)
    return renamed_lines

# Example usage:

if __name__ == "__main__":
   
    
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    renamed_lines = rename_nodes(lines)
    print(renamed_lines)
    result = sort_tensors_with_values(renamed_lines)
    print(result)
