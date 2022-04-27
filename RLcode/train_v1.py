from types import prepare_class
import torch
import numpy as np
from tqdm import tqdm
import pickle
from random import sample
from traffic_simulator import TrafficSim
from os import listdir
from modelFCN import Policynet, Valuenet, Discriminator, Qnet
from torch.nn.functional import binary_cross_entropy_with_logits,mse_loss,softplus
import matplotlib.pyplot as plt

def target(expert_pro, generator_pro):
    m = expert_pro.shape[0]
    E = torch.sum(torch.log(generator_pro) + torch.log(1 - expert_pro)) / m
    return E

def merge_obs(obs):
    '''
    pos  speed vel acc a_vel a_acc 
    3 + 1 +  3 + 3 + 1 = 17
    '''
    def merge(obs_):
        position, speed,vel, acc, a_vel,a_acc,heading = \
            obs_.position ,obs_.speed, obs_.linear_velocity, obs_.linear_acceleration,\
                 obs_.angular_velocity, obs_.angular_acceleration, obs_.heading
        position = position.reshape(1,-1)
        speed = np.array(speed).reshape(1,1)
        vel = vel.reshape(1,-1)
        acc = acc.reshape(1,-1)
        heading = np.array(heading).reshape(1,-1)

        a_vel = a_vel.reshape(1,-1)
        a_acc = a_acc.reshape(1,-1)
        
        return np.concatenate([position, speed, heading, vel, acc, a_vel,a_acc],axis =1)

    ego_obs = obs.ego_vehicle_state
    ego_pos = ego_obs.position
    neibor_obs = obs.neighborhood_vehicle_states
    tmp = list(map(lambda x : (x,np.sum((x.position - ego_pos)**2)),neibor_obs))
    tmp = sorted(tmp, key=lambda x: x[1])

    ego_part = merge(ego_obs)

    candidate = [ego_part]
    num = 0
    for neibor in tmp:
        num += 1
        neibor = neibor[0]
        speed = np.array(neibor.speed).reshape(1,-1)
        position = neibor.position.reshape(1,-1)
        heading = np.array(neibor.heading).reshape(1,-1)
        neibor_state = np.concatenate([position,speed,heading],axis = 1)
        candidate.append(neibor_state)
        if num == 10:
             break
    if num < 10:
        candidate.append(np.zeros(5*(10-num)).reshape(1,-1))
    
    state = np.concatenate(candidate,axis = 1)     
    return state

def data_loader(path):
    files = listdir(path)
    total_state = []
    total_actions = []
    total_next_state = []
    total_terminal = []

    for f in files:
        with open(path + '/{}'.format(f) ,'rb') as t:
            expert = pickle.load(t)
    
        exp_state = expert['observations']
        exp_actions = expert['actions']
        exp_terminal = expert['terminals']
        exp_next_state = expert['next_observations']
    
        exp_state = np.concatenate(list(map(lambda x:np.array(x).squeeze(1),exp_state)))
        exp_actions = np.concatenate(exp_actions)
        exp_terminal = np.concatenate(exp_terminal)
        exp_next_state = np.concatenate(list(map(lambda x:np.array(x).squeeze(1),exp_next_state)))

        total_state.append(exp_state)
        total_actions.append(exp_actions)
        total_next_state.append(exp_next_state)
        total_terminal.append(exp_terminal)
        #break #............................................................................................jideshan
    total_state = np.concatenate(total_state)
    total_actions = np.concatenate(total_actions)
    total_next_state = np.concatenate(total_next_state)
    total_terminal = np.concatenate(total_terminal)

    return total_state,total_actions,total_next_state,total_terminal

def Sample(state,next_state,act,size):
    length = state.shape[0]
    per = np.random.permutation(length)[:size]
    sample_state = state[per]
    sample_next_state = next_state[per]
    sample_act = act[per]
    return  sample_state, sample_next_state, sample_act

def clip(act):
    if act[:,0] > 2.5:
        act[:,0] = 2.5
    elif act[:,0] < -2.5:
        act[:,0] = -2.5
    if act[:,1] > 0.5:
        act[:,1] = 0.5
    elif act[:,1] < -0.5:
        act[:,1] = -0.5
    return act

if __name__ == '__main__':
    path = './expert'
    expert_state,expert_act,expert_next_state,expert_terminal = \
        data_loader(path)
    device = torch.device('cpu')

    lr_d = 0.0003
    lr = 0.005
    lr_q = 0.02
    gamma = 0.99
    grad_weight = 10
    log_alpha = torch.tensor(np.log(0.2),requires_grad= True)
    alpha = 0.2

    epsilon_begin = 0.3
    epsilon_over = 0.001
    epsilon_decay = 100
    rou = 0.95
    dis_threshold = 0.6
    gen_threshold = 0.5

    ep_episode = lambda x : epsilon_over + (epsilon_begin - epsilon_over) * np.exp(
        -1. * x / epsilon_decay)
    dis_loss = []
    gen_loss = []
    critic_loss = []
    q_value = []
    dists = []

    state_dim = 67
    gen = Policynet(state_dim,2)
    dis = Discriminator(state_dim,2)
    #val = Valuenet(40)
    qnet1 = Qnet(state_dim,2)  
    qnet2 = Qnet(state_dim,2)
    qtarget1 = Qnet(state_dim,2)
    qtarget2 = Qnet(state_dim,2)

    # gen.load_state_dict(torch.load('./result/param/gen_param.pth'))
    # dis.load_state_dict(torch.load('./result/param/dis_param.pth'))
    # qnet1.load_state_dict(torch.load('./result/param/qnet_1_param.pth'))
    # qnet2.load_state_dict(torch.load('./result/param/qnet_2_param.pth'))
    # # #val.load_state_dict(torch.load('./result/param/val_param.pth'))
    with torch.no_grad():
        for p,p_targ in zip(qnet1.parameters(),qtarget1.parameters()):
           p_targ = p
        for p,p_targ in zip(qnet2.parameters(),qtarget2.parameters()):
            p_targ = p
    for p1,p2 in zip(qtarget1.parameters(),qtarget2.parameters()):
        p1.requires_grad = False
        p2.requires_grad = False
    ego_replay_buffer = []

    # opt_dis = torch.optim.SGD(dis.parameters(),lr=lr)
    opt_dis = torch.optim.Adam(dis.parameters(),lr=lr_d,betas=(0, 0.99))
    opt_gen = torch.optim.Adam(gen.parameters(),lr=lr,betas=(0.25, 0.99))
    opt_q_1 = torch.optim.Adam(qnet1.parameters(),lr=lr_q,betas=(0.25, 0.99))
    opt_q_2 = torch.optim.Adam(qnet2.parameters(),lr=lr_q,betas=(0.25, 0.99))
    opt_alpha = torch.optim.Adam([log_alpha],lr=lr_q,betas=(0.25, 0.99)) 

    batch_size = 256
    episode = 50 #10
    n_dist = 1000
    start_std = 1
    over_std = 0.002
    eps_std = lambda x : (start_std - over_std)*np.exp(-x/10000) + over_std

    exp_score = []
    self_score = []
    min_size = 20000
    max_size = 800000
    sample_size = 10000
    add_size = 5000
    update_num = 100
    update_step = 0

    env = TrafficSim(["./ngsim"])
    obs = env.reset()
    state = merge_obs(obs)
    prev_state = state

    eps = 0
    step_num = 0
    avg_dist = []
    avg_rew = []
    print('Begin Train ')
    gen.eval()
    finish_train = False
    epsilon = ep_episode(eps) 
    for eps in tqdm(range(episode)): 
        epsilon = ep_episode(eps)
        print('Sampling...')
        with torch.no_grad():
            while step_num < min_size or step_num % add_size != 0 :
                if finish_train:
                    gen.eval()
                    obs = env.reset()
                    state = merge_obs(obs)
                    prev_state = state
                    finish_train = False
                policy = gen(torch.FloatTensor(state))
                rd = np.random.rand(1)
                if rd > epsilon:
                    act = policy.rsample().detach().cpu().numpy()
                    act = clip(act)

                else:
                    act = np.random.normal(-2.5, 2.5, size=(1,2))
                    act = clip(act)
                act = act.reshape(2,1)
                obs,rew,done,info = env.step(act)
                avg_rew.append(rew)
                
                state = merge_obs(obs)
                step_sample = [prev_state.reshape(1,-1),state.reshape(1,-1),act.T,[done]]
                ego_replay_buffer.append(step_sample)
                if done:
                    avg_dist.append(info)
                    obs = env.reset()
                    state = merge_obs(obs)
                prev_state = state
                step_num += 1
        print("Finish Sampling")  


        print('Train in episode {}'.format(eps))
        if step_num > max_size:
            step_num = max_size // 2
            ego_replay_buffer = ego_replay_buffer[:max_size // 2]

        print('train')
            # sample from experince pool
        expert_train_state, expert_train_next_state,expert_train_act = \
            Sample(expert_state,expert_next_state,expert_act,sample_size)

        ego_train_state,ego_train_next_state, ego_train_act,ego_train_rew, ego_train_done = zip(*sample(ego_replay_buffer, sample_size))
            
        ego_train_state, ego_train_next_state, ego_train_act,ego_train_rew,ego_train_done = \
                np.concatenate(ego_train_state),np.concatenate(ego_train_next_state),np.concatenate(ego_train_act),np.array(ego_train_rew),np.concatenate(ego_train_done)
        expert_train_state, expert_train_next_state,expert_train_act = \
                torch.FloatTensor(expert_train_state), torch.FloatTensor(expert_train_next_state), torch.FloatTensor(expert_train_act)

        iter_num = np.int32(np.ceil(sample_size / batch_size))

        for i in range(update_num):
            dis.train()
            opt_dis.zero_grad()
            expert_train_state, _,expert_train_act = \
                Sample(expert_state,expert_next_state,expert_act,batch_size)

            ego_train_state,_, ego_train_act, _ = zip(*sample(ego_replay_buffer,batch_size))
            
            ego_train_state,  ego_train_act = np.concatenate(ego_train_state),np.concatenate(ego_train_act)
            expert_train_state,expert_train_act = torch.FloatTensor(expert_train_state),  torch.FloatTensor(expert_train_act)

            ego_socre = dis.cal_pro(ego_train_state,ego_train_act)
            expert_score = dis.cal(expert_train_state,expert_train_act)

            ego_label, expert_label = torch.zeros_like(ego_score), torch.ones_like(expert_score)
            loss_dis = binary_cross_entropy_with_logits(expert_score, expert_label)\
                         + binary_cross_entropy_with_logits(ego_score, ego_label)
            ratio = torch.tensor(np.random.random(ego_train_act.shape[0])).reshape(-1,1)
            cat_state = ratio * ego_train_state + (1-ratio)* expert_train_state
            cat_act = ratio * ego_train_act + (1-ratio)* expert_train_act
            
            cat_state,cat_act = cat_state.float().detach(), cat_act.float().detach()
            cat_state.requires_grad_(True)
            cat_act.requires_grad_(True)

            grad = torch.autograd.grad(
                outputs=dis.cal_pro(cat_state,cat_act).sum(),
                inputs=[cat_state,cat_act],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            loss_dis_pen = grad_penalty * grad_weight
            loss_dis += loss_dis_pen
            loss_dis.backward()
            opt_dis.step()
            dis_loss.append(loss_dis.item())
            exp_score.append(expert_score.mean().item())
            self_score.append(ego_score.mean().item())

        for i in range(update_num):
            opt_gen.zero_grad()
            opt_q_1.zero_grad()
            opt_q_2.zero_grad()
            opt_alpha.zero_grad()
            gen.train()
            dis.eval()

            # expert_train_state, expert_train_next_state,expert_train_act = \
            #         Sample(expert_state,expert_next_state,expert_act,batch_size)

            ego_train_state,ego_train_next_state, ego_train_act, ego_train_done = zip(*sample(ego_replay_buffer, batch_size))
            
            ego_train_state, ego_train_next_state, ego_train_act, ego_train_done = \
                np.concatenate(ego_train_state),np.concatenate(ego_train_next_state),np.concatenate(ego_train_act),np.concatenate(ego_train_done)
            # expert_train_state, expert_train_next_state,expert_train_act = \
            #     torch.FloatTensor(expert_train_state), torch.FloatTensor(expert_train_next_state), torch.FloatTensor(expert_train_act)            
            ego_rew = softplus(torch.log(dis.cal_pro(ego_train_state,ego_train_act)),beta = 1).detach()
            next_act_dist = gen(ego_train_next_state)
            next_act = next_act_dist.sample().detach()
            next_act_log_pro = torch.sum(next_act_dist.log_prob(next_act).detach())
            next_act_log_pro -= torch.sum(2*(np.log(2) - next_act - softplus(-2*next_act)))

            q_target_val_1 = qtarget1.cal_q_val(ego_train_next_state,next_act)
            q_target_val_2 = qtarget2.cal_q_val(ego_train_next_state,next_act)
            q_val = torch.min(q_target_val_1,q_target_val_2)
            q_target = ego_rew.reshape(-1,)  + gamma * (1-ego_train_done) * (q_val- alpha * next_act_log_pro) #
            q_target = q_target.detach().unsqueeze(1)
            
            q_val_1 = qnet1.cal_q_val(ego_train_state,ego_train_act)
            q_val_2 = qnet2.cal_q_val(ego_train_state,ego_train_act)
            loss_q1 = mse_loss(q_val_1,q_target)#smooth_l1_loss(q_target,q_val_1)
            loss_q2 = mse_loss(q_val_2,q_target)#smooth_l1_loss(q_target,q_val_2)
            loss_q = loss_q1 + loss_q2
            loss_q.backward()
            critic_loss.append(loss_q.item())
            opt_q_1.step()
            opt_q_2.step()

            policy = gen(ego_train_state)
            new_act = policy.rsample()
            new_ego_pro = torch.sum(policy.log_prob(new_act))#new_act
            new_ego_pro -= torch.sum(2*(np.log(2) - new_act - softplus(-2*new_act)))
            q_new_val_1 = qnet1.cal_q_val(ego_train_state,new_act).detach()
            q_new_val_2 = qnet2.cal_q_val(ego_train_state,new_act).detach()
            
            q_new_val = torch.min(q_new_val_1,q_new_val_2)
            q_new_val = q_new_val.detach()

            loss_gen = -torch.mean(q_new_val - new_ego_pro*alpha)# + binary_cross_entropy_with_logits(bat_ego_score.detach(), torch.ones_like(bat_ego_score)) * bat_ego_pro)
            loss_gen.backward()
            opt_gen.step()
            gen_loss.append(loss_gen.item())
            q_value.append(torch.mean(q_new_val).item())

            log_prob = new_ego_pro.detach() + 2
            alpha_loss = -(log_alpha * log_prob).mean()
            alpha_loss.backward()
            opt_alpha.step()
            alpha = np.exp(log_alpha.detach().cpu().numpy())

            with torch.no_grad():
                for p, p_targ in zip(qnet1.parameters(), qtarget1.parameters()):
                    p_targ.data = p_targ.data.mul_(rou)
                    p_targ.data = p_targ.data.add_((1 - rou) * p.data)
                for p, p_targ in zip(qnet2.parameters(), qtarget2.parameters()):
                    p_targ.data = p_targ.data.mul_(rou)
                    p_targ.data = p_targ.data.add_((1 - rou) * p.data)

        step_num += 1
        for i in range(iter_num):
            noise_std = eps_std(update_step)
            update_step += 1
            opt_dis.zero_grad()
            opt_gen.zero_grad()
            opt_q_1.zero_grad()
            opt_q_2.zero_grad()
            opt_alpha.zero_grad()
            bat_exp_state = expert_train_state[i*batch_size:(i+1)*batch_size].reshape(-1,state_dim)
            bat_exp_act = expert_train_act[i*batch_size:(i+1)*batch_size].reshape(-1,2)

            bat_exp_state += noise_std * torch.randn(bat_exp_state.shape)
            bat_exp_act += noise_std * torch.randn(bat_exp_act.shape)

            bat_ego_state = torch.FloatTensor(ego_train_state[i*batch_size:(i+1)*batch_size]).reshape(-1,state_dim)
            bat_ego_state += noise_std * torch.randn(bat_ego_state.shape)
            
            bat_ego_next_state = torch.FloatTensor(ego_train_next_state[i*batch_size:(i+1)*batch_size]).reshape(-1,state_dim)
            bat_ego_act = torch.FloatTensor(ego_train_act[i*batch_size:(i+1)*batch_size]).reshape(-1,2)
            bat_ego_act += noise_std * torch.randn(bat_ego_act.shape)
            # bat_ego_rew = torch.FloatTensor(ego_train_rew[i*batch_size:(i+1)*batch_size])

            bat_ego_done = torch.FloatTensor(ego_train_done[i*batch_size:(i+1)*batch_size])
############################################################################################################################################################################################################
            bat_ego_score = dis.cal_pro(bat_ego_state,bat_ego_act)
            bat_expert_score = dis.cal_pro(bat_exp_state,bat_exp_act)
            loss_dis = binary_cross_entropy_with_logits(bat_expert_score, torch.ones_like(bat_expert_score))\
                         + binary_cross_entropy_with_logits(bat_ego_score, torch.zeros_like(bat_ego_score))
            # Add gradient Penalty
            ratio = torch.tensor(np.random.random(bat_ego_act.shape[0])).reshape(-1,1)
            cat_state = ratio * bat_ego_state + (1-ratio)* bat_exp_state
            cat_act = ratio * bat_ego_act + (1-ratio)* bat_exp_act
            
            cat_state,cat_act = cat_state.float().detach(), cat_act.float().detach()
            cat_state.requires_grad_(True)
            cat_act.requires_grad_(True)

            grad = torch.autograd.grad(
                outputs=dis.cal_pro(cat_state,cat_act).sum(),
                inputs=[cat_state,cat_act],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            loss_dis_pen = grad_penalty * grad_weight
            loss_dis += loss_dis_pen
            loss_dis.backward()
            opt_dis.step()
            dis_loss.append(loss_dis.item())
                # if loss_dis.item() > dis_threshold:
                #     loss_dis.requires_grad_(True) 
                #     loss_dis.backward()
                # loss_dis.requires_grad_(True) 
                #  Hold the dis para
                # loss_val = smooth_l1_loss(ego_score + gamma * value_next, value)
 ########################################################################################################################################################################################################           
            next_act_dist = gen(bat_ego_next_state)
            next_act = next_act_dist.sample().detach()
            next_act_log_pro = torch.sum(next_act_dist.log_prob(next_act).detach())
            next_act_log_pro -= torch.sum(2*(np.log(2) - next_act - softplus(-2*next_act)))
            dis.eval()
            bat_ego_rew = softplus(torch.log(dis.cal_pro(bat_ego_state,bat_ego_act)),beta = 1).detach()
            dis.train()
            with torch.no_grad():
                q_target_val_1 = qtarget1.cal_q_val(bat_ego_next_state,next_act)
                q_target_val_2 = qtarget2.cal_q_val(bat_ego_next_state,next_act)
                q_val = torch.cat([q_target_val_1.reshape(-1,1),q_target_val_2.reshape(-1,1)],dim = 1)
                q_val,_ = torch.min(q_val,dim = 1)
                q_target = bat_ego_rew.reshape(-1,)  + gamma * (1-bat_ego_done) * (q_val- alpha * next_act_log_pro) #
                q_target = q_target.detach().unsqueeze(1)
            
            q_val_1 = qnet1.cal_q_val(bat_ego_state,bat_ego_act)
            q_val_2 = qnet2.cal_q_val(bat_ego_state,bat_ego_act)
            loss_q1 = mse_loss(q_val_1,q_target)#smooth_l1_loss(q_target,q_val_1)
            loss_q2 = mse_loss(q_val_2,q_target)#smooth_l1_loss(q_target,q_val_2)
            loss_q = loss_q1 + loss_q2
            loss_q.backward()
            critic_loss.append(loss_q.item())
            opt_q_1.step()
            opt_q_2.step()
##############################################################################################################################################################################################################
                # for p1,p2 in zip(qnet1.parameters(),qnet2.parameters()):
                #     p1.requires_grad=False
                #     p2.requires_grad=False  #Hold Qnet Parameters

            policy = gen(bat_ego_state)
            new_act = policy.rsample()
            bat_new_ego_pro = torch.sum(policy.log_prob(new_act))#new_act
            bat_new_ego_pro -= torch.sum(2*(np.log(2) - new_act - softplus(-2*new_act)))
            q_new_val_1 = qnet1.cal_q_val(bat_ego_state,new_act).detach()
            q_new_val_2 = qnet2.cal_q_val(bat_ego_state,new_act).detach()
            
            q_new_val,_ = torch.min(torch.cat([q_new_val_1,q_new_val_2],dim = 1),dim = 1)
            q_new_val = q_new_val.detach()

            loss_gen = -torch.mean(q_new_val - bat_new_ego_pro*alpha)# + binary_cross_entropy_with_logits(bat_ego_score.detach(), torch.ones_like(bat_ego_score)) * bat_ego_pro)
            loss_gen.backward()
            opt_gen.step()
            gen_loss.append(loss_gen.item())
            q_value.append(torch.mean(q_new_val).item())
############################################################################################################################################################################################################
            # update alpha
            log_prob = bat_new_ego_pro.detach() + 2
            alpha_loss = -(log_alpha * log_prob).mean()
            alpha_loss.backward()
            opt_alpha.step()
            alpha = np.exp(log_alpha.detach().cpu().numpy())

###########################################################################################################################################################################################################
            # if loss_gen.item() > gen_threshold:
            #     loss_gen.backward()
            #     opt_gen.step()
            if update_step % update == 0:
                with torch.no_grad():
                    for p, p_targ in zip(qnet1.parameters(), qtarget1.parameters()):
                        p_targ.data = p_targ.data.mul_(rou)
                        p_targ.data = p_targ.data.add_((1 - rou) * p.data)
                    for p, p_targ in zip(qnet2.parameters(), qtarget2.parameters()):
                        p_targ.data = p_targ.data.mul_(rou)
                        p_targ.data = p_targ.data.add_((1 - rou) * p.data)

            expert_score = bat_expert_score.detach().cpu().numpy().mean()
            ego_score = bat_ego_score.detach().cpu().numpy().mean()
            exp_score.append(expert_score)
            self_score.append(ego_score)
            step_num += 1
        print('Finish episode {}'.format(eps))
        finish_train = True
        torch.save(gen.state_dict(),'./result/param/gen_param.pth')
        torch.save(dis.state_dict(),'./result/param/dis_param.pth')
        torch.save(qnet1.state_dict(),'./result/param/qnet_1_param.pth')
        torch.save(qnet2.state_dict(),'./result/param/qnet_2_param.pth')
            # with open('./result/param/rec.txt','w') as f:
            #     f.write('{}'.format(eps))
            #     f.write('\n')
            #     f.write(exp_score)
            #     f.write('\n')
            #     f.write(ego_score)
        print('Sampling...')
        dists.append(np.mean(avg_dist))
        rews.append(np.mean(avg_rew))
        avg_dist = []
        avg_rew = []

    env.destroy()
    plt.plot(exp_score)
    plt.savefig('./result/plot/expert_score.jpg')
    plt.plot(self_score)
    plt.savefig('./result/plot/self_score.jpg')
    plt.close()

    plt.plot(dis_loss)
    plt.savefig('./result/plot/discrimiate.jpg')
    plt.plot(gen_loss)
    plt.savefig('./result/plot/generator.jpg')
    plt.close()
    plt.plot(critic_loss)
    plt.savefig('./result/plot/critic.jpg')
    plt.close()
    plt.plot(rews)
    plt.savefig('./result/plot/rewards.jpg')
    plt.close()
    plt.plot(dists)
    plt.savefig('./result/plot/dists.jpg')
    plt.close()
