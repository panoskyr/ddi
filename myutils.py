import torch

# saves to a txt file afeture in the form of node_number:feature
def save_to_txt(filename, data):
    with open(filename+".txt", "w") as myfile:
        for key, value in sorted(data.items()):
            myfile.write(str(key) + ' ' + str(value) + '\n')



def get_dict_from_file(filename):
    d=dict()
    with open(filename, 'r') as f:
        for line in f:
            (key,val)=line.split()
            val=float(val)
            #print(val,type(val))
            d[int(key)]=val
    return d

def save_tensor_to_file(filename, data):
    torch.save(data, filename+".pt")
