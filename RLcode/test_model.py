from traffic_simulator import TrafficSim
import torch
import numpy as np
from modelFCN import Policynet

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

def clip(act):
    if act[0] > 2.5:
        act[0] = -2.5
    elif act[0] < -2.5:
        act[0] = -2.5
    if act[1] > 0.5:
        act[1] = 0.5
    elif act[1] < -0.5:
        act[1] = -0.5
    return act

if __name__ == '__main__':
    policy = Policynet(67,2)
    with open("./result/param/rec.txt") as f:
            eps_rec = int(f.readline())
            alpha_rec = np.double(f.readline())
            state_mean = []
            state_std = []
            tmp = f.readline()
            while tmp != 'next\n':
                state_mean.append(float(tmp))
                tmp = f.readline()
            tmp = f.readline()
            while tmp != 'next\n':
                state_std.append(float(tmp))
                tmp = f.readline()

    state_mean = np.array(state_mean)
    state_std = np.array(state_std)

    policy.load_state_dict(torch.load('./result/param/gen_param.pth'))
    policy.eval()
    max_car = 100
    total_car = 0
    success_car = 0

    env = TrafficSim(["./ngsim"])
    obs = env.reset()
    state = torch.FloatTensor((merge_obs(obs)-state_mean)/(state_std+1e-8))
    score = 0
    step = 0
    with torch.no_grad():
        while True:
            if total_car == max_car:
                break
            action = policy(state,True)
            action = action.detach().cpu().numpy().squeeze(0)
            action = clip(action)
            obs,_,done,info = env.step(action)

            # TODO ADD end condition and handle the form of input 
            if done:
                print('{} is done'.format(total_car))
                total_car += 1
                success = obs.events.reached_goal
                if success:
                    success_car += 1
                obs = env.reset()
                score += info
                print(info)
            state = torch.FloatTensor((merge_obs(obs)-state_mean)/(state_std+1e-8))

    print(score / total_car)
    print(success_car / total_car)
    env.destroy()
    # print('acc is {}'.format(success_car / total_car))
