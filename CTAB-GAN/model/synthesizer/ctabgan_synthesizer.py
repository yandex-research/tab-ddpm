import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from tqdm import tqdm


class Classifier(Module):
    def __init__(self,input_dim, dis_dims,st_ed):
        super(Classifier,self).__init__()
        dim = input_dim-(st_ed[1]-st_ed[0])
        seq = []
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        
        if (st_ed[1]-st_ed[0])==1:
            seq += [Linear(dim, 1)]
        
        elif (st_ed[1]-st_ed[0])==2:
            seq += [Linear(dim, 1),Sigmoid()]
        else:
            seq += [Linear(dim,(st_ed[1]-st_ed[0]))] 
        
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        label=None
        
        if (self.str_end[1]-self.str_end[0])==1:
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        if ((self.str_end[1]-self.str_end[0])==2) | ((self.str_end[1]-self.str_end[0])==1):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)

def get_st_ed(target_col_index,output_info):
    st = 0
    c= 0
    tc= 0
    for item in output_info:
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1
        tc+=1    
    ed= st+output_info[tc][0] 
    return (st,ed)

def random_choice_prob_index_sampling(probs,col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval

class Cond(object):
    def __init__(self, data, output_info):
       
        self.model = []
        st = 0
        counter = 0
        for item in output_info:
           
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0  
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))  
        self.p_sampling = []
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':            
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)  
                tmp_sampling = np.sum(data[:, st:ed], axis=0)     
                tmp = np.log(tmp + 1)  
                tmp = tmp / np.sum(tmp)
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
                
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        opt1prime = random_choice_prob_index(self.p[idx]) 
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
      
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec

def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            continue

        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]

class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
                
    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]

class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        info = len(layers)-2
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)

def determine_layers_disc(side, num_channels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    print()
    layers_D += [

        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), 
        Sigmoid() 
    ]
    
    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    return layers_G


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

class CTABGANSynthesizer:
    def __init__(self,
                 lr=2e-4,
                 class_dim=(256, 256, 256, 256),
                 random_dim=128,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=1024,
                 epochs=1,
                 device=torch.device("cpu")):
                 

        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, type={}, no_train=False):
        print("Fit started.")
        problem_type = None
        target_index=None
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed)
        self.transformer.fit() 
        
        train_data = self.transformer.transform(train_data.values)
        
        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)
        		
        sides = [4, 8, 16, 24, 32]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        
        sides = [4, 8, 16, 24, 32]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
        
        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        
        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)
        optimizer_params = dict(lr=self.lr, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier=None
        optimizerC= None
        if target_index != None:
            st_ed= get_st_ed(target_index,self.transformer.output_info)
            classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)
        
        
        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)
        
        
        if no_train: return

        print("Training started.")
        for i in range(self.epochs):
            # for _ in range(steps_per_epoch):
                
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample_train(self.batch_size)

            c, m, col, opt = condvec
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            perm = np.arange(self.batch_size)
            np.random.shuffle(perm)
            real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
            c_perm = c[perm]
                
            real = torch.from_numpy(real.astype('float32')).to(self.device)
                
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, self.transformer.output_info)
                
            fake_cat = torch.cat([fakeact, c], dim=1)
            real_cat = torch.cat([real, c_perm], dim=1)
                
            real_cat_d = self.Dtransformer.transform(real_cat)
            fake_cat_d = self.Dtransformer.transform(fake_cat)
                
            optimizerD.zero_grad()
            y_real,_ = discriminator(real_cat_d)
            y_fake,_ = discriminator(fake_cat_d)
            loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
            loss_d.backward()
            optimizerD.step()
                
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            
            condvec = self.cond_generator.sample_train(self.batch_size)

            c, m, col, opt = condvec
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

            optimizerG.zero_grad()

            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, self.transformer.output_info)

            fake_cat = torch.cat([fakeact, c], dim=1) 
            fake_cat = self.Dtransformer.transform(fake_cat)
                
            y_fake,info_fake = discriminator(fake_cat)
            
            cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)

            _,info_real = discriminator(real_cat_d)
            
            g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
            g.backward(retain_graph=True)
            loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
            loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
            loss_info = loss_mean + loss_std 
            loss_info.backward()
            optimizerG.step()

            if (i + 1) % 500 == 0:
                print(f"Step: {i}/{self.epochs} Loss: {loss_mean:.4f}")

            if problem_type:
                        
                fake = self.generator(noisez)
                
                faket = self.Gtransformer.inverse_transform(fake)
                
                fakeact = apply_activate(faket, self.transformer.output_info)
                
                real_pre, real_label = classifier(real)
                fake_pre, fake_label = classifier(fakeact)
                    
                c_loss = CrossEntropyLoss() 
                
                if (st_ed[1] - st_ed[0])==1:
                    c_loss= SmoothL1Loss()
                    real_label = real_label.type_as(real_pre)
                    fake_label = fake_label.type_as(fake_pre)
                    real_label = torch.reshape(real_label,real_pre.size())
                    fake_label = torch.reshape(fake_label,fake_pre.size())
                    
                
                elif (st_ed[1] - st_ed[0])==2:
                    c_loss = BCELoss()
                    real_label = real_label.type_as(real_pre)
                    fake_label = fake_label.type_as(fake_pre)

                loss_cc = c_loss(real_pre, real_label)
                loss_cg = c_loss(fake_pre, fake_label)

                optimizerG.zero_grad()
                loss_cg.backward()
                optimizerG.step()

                optimizerC.zero_grad()
                loss_cc.backward()
                optimizerC.step()
                                
    @torch.no_grad()
    def sample(self, n, seed=0):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        sample_batch_size = 8092
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // sample_batch_size + 1
        
        data = []
        
        for i in range(steps):
            noisez = torch.randn(sample_batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(sample_batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(sample_batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket,output_info)
            # print(len(data))
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result = self.transformer.inverse_transform(data)
        
        return result[0:n]

