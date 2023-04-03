import torch

def one_hot_cloud(data, num_labels, position):
    data = data.cpu()
    position = position.cpu()
    encoded_tensor =  torch.zeros(5,5,2048)
    
    for i in range (data.shape[0]):
        zeros = torch.zeros((max(data.shape), num_labels))
        zeros[:,position[i]] = 1
        encoded_data = torch.cat((zeros, data[0].T), dim=1).T
        encoded_tensor[i] = encoded_data
    
    encoded_tensor = encoded_tensor.to('cuda')
    return encoded_tensor

def generate():

    output =[]
    my_dict = {'p_prior_mus':[], 'p_prior_logvars':[], 'p_prior_samples':[]}

    for k in range(4):
        for key in my_dict:
            a = torch.rand(5, 3, 2048)
            b = torch.rand(33, 5, 5, 2048)
            b[:,:,0:2,:] = 0
            c = []
            c.append(a)
            for i in range (b.shape[0]):
                c.append(b[i])
            my_dict[key] = c
        output.append(dict)
    return output

output_decoder = generate()

output_decoder[0]['p_prior_logvars'][0] = one_hot_cloud(output_decoder[0]['p_prior_logvars'][0], 2, cloud_labels)
output_decoder[0]['p_prior_mus'][0] = one_hot_cloud(output_decoder[0]['p_prior_mus'][0], 2, cloud_labels)

check = []
for k in range(len(output_decoder)):
    for key in (output_decoder[k]):
        for i in range (len(output_decoder[k][key])):
                check.append(output_decoder[k][key][i].shape)
        
checked_indices = []
for i in range(len(check)):
    if check[i] == check[101]:
        checked_indices.append(i)

print("debugging")