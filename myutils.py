# saves to a txt file afeture in the form of node_number:feature
def save_to_txt(filename, data):
    with open(filename+".txt", "w") as myfile:
        for key, value in sorted(data.items()):
            myfile.write(str(key) + ' ' + str(value) + '\n')



def get_dict_from_file(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    data = {}
    for line in content:
        key,value= line.split()
        data[int(key)] = value
    return data