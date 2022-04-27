import copy
import numpy as np
import pickle
import argparse
import gc
import psutil

from example_adapter import get_observation_adapter

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType

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

def acceleration_count(obs, obs_next, acc_dict, ang_v_dict, avg_dis_dict):
    acc_dict = {}
    for car in obs.keys():
        car_state = obs[car].ego_vehicle_state
        angular_velocity = car_state.yaw_rate
        ang_v_dict.append(angular_velocity)
        dis_cal = car_state.speed * 0.1
        if car in avg_dis_dict:
            avg_dis_dict[car] += dis_cal
        else:
            avg_dis_dict[car] = dis_cal
        if car not in obs_next.keys():
            continue
        car_next_state = obs_next[car].ego_vehicle_state
        acc_cal = (car_next_state.speed - car_state.speed) / 0.1
        acc_dict.append(acc_cal)


def cal_action(obs, obs_next, dt=0.1):
    act = {}
    for car in obs.keys():
        if car not in obs_next.keys():
            continue
        car_state = obs[car].ego_vehicle_state
        car_next_state = obs_next[car].ego_vehicle_state
        acceleration = (car_next_state.speed - car_state.speed) / dt
        angular_velocity = car_state.yaw_rate
        act[car] = np.array([acceleration, angular_velocity])
    return act


def main(scenario, obs_stack_size=1):
    """Collect expert observations.

    Each input scenario is associated with some trajectory files. These trajectories
    will be replayed on SMARTS and observations of each vehicle will be collected and
    stored in a dict.

    Args:
        scenarios: A string of the path to scenarios to be processed.

    Returns:
        A dict in the form of {"observation": [...], "next_observation": [...], "done": [...]}.
    """
    mem = psutil.virtual_memory()

    agent_spec = AgentSpec(
        interface=AgentInterface(
            max_episode_steps=None,
            waypoints=False,
            neighborhood_vehicles=True,
            ogm=False,
            rgb=False,
            lidar=False,
            action=ActionSpaceType.Imitation,
        ),
        observation_adapter=get_observation_adapter(obs_stack_size),
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenario],
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    expert_obs = []
    expert_acts = []
    expert_obs_next = []
    expert_terminals = []
    
    cars_obs = {}
    cars_act = {}
    cars_obs_next = {}
    cars_terminals = {}

    car_list = []

    prev_vehicles = set()
    done_vehicles = set()
    prev_obs = None
    num_pickle = 0
    while True:
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles

        for v in done_vehicles:
            v = f"Agent-{v}"
            cars_terminals[v][-1] = True
            cars_obs_next[v] = cars_obs[v][1:]
            cars_obs[v] = cars_obs[v][:-1]
            cars_act[v] = np.array(cars_act[v])
            cars_terminals[v] = np.array(cars_terminals[v][:-1])
            expert_obs.append(cars_obs[v])
            expert_acts.append(cars_act[v])
            expert_obs_next.append(cars_obs_next[v])
            expert_terminals.append(cars_terminals[v])
            car_list.append(v)
            if len(car_list) == 100 :
                    with open("./expert/expert_{}.pkl".format(num_pickle), "wb") as f:
                        pickle.dump(
                            {
                             "observations": expert_obs,
                             "actions": expert_acts,
                             "next_observations": expert_obs_next,
                             "terminals": expert_terminals,
                             },
                             f,)
                    num_pickle += 1
                    expert_obs = []
                    expert_acts = []
                    expert_obs_next = []
                    expert_terminals = []

                    final_car = list(set(cars_act.keys()) - set(car_list))
                    car_list = []

                    new_cars_teminal = {}
                    new_cars_obs = {}
                    new_cars_acts = {}

                    for car in final_car:
                        new_cars_teminal[car] = cars_terminals[car]
                        new_cars_acts[car] = cars_act[car]
                        new_cars_obs [car] = cars_obs[car]
                    del cars_obs,cars_act,cars_terminals,cars_obs_next
                    gc.collect()

                    cars_obs = copy.copy(new_cars_obs)
                    cars_act = copy.copy(new_cars_acts)
                    cars_terminals = copy.copy(new_cars_teminal)
                    cars_obs_next = {}
                    del new_cars_teminal,new_cars_acts,new_cars_obs
                    gc.collect()
                    print('packet {} finish!!!!!!!!!!!!!!'.format(num_pickle))
                    print('Free mem {}'.format(float(mem.free)))

            print(f"{v} Ended")
        # if num_pickle == 1:
        #     break

        if len(current_vehicles) == 0:
            break

        smarts.attach_sensors_to_vehicles(
        agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        obs, _, _, dones = smarts.observe_from(
            smarts.vehicle_index.social_vehicle_ids()
        )


        # handle actions
        if prev_obs is not None:
            act = cal_action(prev_obs, obs)
            for car in act.keys():
                if cars_act.__contains__(car):
                    cars_act[car].append(act[car])
                else:
                    cars_act[car] = [act[car]]
        prev_obs = copy.copy(obs)

        # handle observations
        cars = obs.keys()
        for car in cars:
            _obs = agent_spec.observation_adapter(obs[car])
            if cars_obs.__contains__(car):
<<<<<<< HEAD
                cars_obs[car].append(merge_obs(obs[car]))
                cars_terminals[car].append(dones[car])
            else:
                cars_obs[car] = [merge_obs(obs[car])]
=======
                cars_obs[car].append(_obs)
                cars_terminals[car].append(dones[car])
            else:
                cars_obs[car] = [_obs]
>>>>>>> refs/remotes/origin/main
                cars_terminals[car] = [dones[car]]

    if len(car_list)> 0:
        with open("./expert/expert_{}.pkl".format(num_pickle), "wb") as f:
            pickle.dump(
                {
                 "observations": expert_obs,
                 "actions": expert_acts,
                 "next_observations": expert_obs_next,
                 "terminals": expert_terminals,
                 },
                 f,)
        print('Final one {}'.format(num_pickle +1))


    # for car in cars_obs:
    #     cars_obs_next[car] = cars_obs[car][1:]
    #     cars_obs[car] = cars_obs[car][:-1]
    #     cars_act[car] = np.array(cars_act[car])
    #     cars_terminals[car] = np.array(cars_terminals[car][:-1])
    #     expert_obs.append(cars_obs[car])
    #     expert_acts.append(cars_act[car])
    #     expert_obs_next.append(cars_obs_next[car])
    #     expert_terminals.append(cars_terminals[car])

    
    print("Sampling Finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    smarts.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="./ngsim",
    )
    args = parser.parse_args()
    main(scenario=args.scenario)
