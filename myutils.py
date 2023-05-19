import torch
import matplotlib.pyplot as plt

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

def draw_metric_per_epoch(metric, title, ylabel, xlabel, filename="fil1"):
    plt.plot(metric)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    plt.savefig(filename+".png")
    plt.close()


def load_data_from_txt(filename):
    with open(filename+".txt", "r") as myfile:
        to_return={}
        for line in myfile:
            (key,value)=line.split()
            to_return[int(key)]=(float(value))
    return to_return

