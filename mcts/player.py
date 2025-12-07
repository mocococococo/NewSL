import time
from copy import deepcopy
import numpy as np
import torch

from mcts.node import Node
from mcts.config import N_SCORE
from mcts.utils import dist_v_to_exp_v, score_to_idx
from nn_mcts.feature import generate_input_planes
from nn_mcts.utility import get_torch_device, load_network
from nn_mcts.network.dual_net import DualNet
from nn_mcts.other import generate_move_from_policy, index_to_shot
from board.constant import BOARD_SIZE, PLANES_SIZE
from dc3client.models import StoneRotation


class Player:
    def __init__(self, ucb_const, pw_const, init_temperature, out_temperature, num_init, num_sample, l, max_depth, use_gpu=True, model_path="sl-model.bin"):
        self.root_node = Node(0.0, 0.0, 0.0, 0.0)
        self.root_env = None
        self.env_dict = None
        self.num_node = None
        self.num_playout = None
        
        self.ucb_const = ucb_const
        self.pw_const = pw_const 
        self.num_init = num_init
        self.device = get_torch_device(use_gpu=use_gpu)
        self.network = load_network(model_path, use_gpu=use_gpu)
        self.network.to(self.device)

        self.init_temperature = init_temperature
        self.out_temperature = out_temperature
        self.num_sample = num_sample 
        self.l = l # control the area of sample space
        self.max_depth = max_depth

        self.kari = 0


    def reset(self, root_env, num_playout):
        #self.root_node = Node(0.131725, 2.39969, StoneRotation.counterclockwise, 0.5)
        self.root_node = Node(0.0, 0.0, 0.0, 0.0)
        self.root_env = root_env
        self.env_dict = {"": deepcopy(self.root_env)}
        self.num_node = 0
        self.num_playout = num_playout

        

    #ゲーム状況stateを受け取り、それに対するMCTSの結果を出力する
    def think(self, root_env, num_playout, max_time = float('inf')):
        st = time.time()
        end_time = st + max_time
        #print("end_time: ", end_time)
        #1.与えられた盤面から根ノードを作り、諸々初期化。(reset())
        self.reset(root_env, num_playout)

        #2.根ノードの盤面から、ニューラルネットワークを用いてポリシーとバリューを予測。(pledict())
        root_prediction_p, root_leaf_black_eval = self.predict(self.root_env)
        root_prediction_p = root_prediction_p.view(-1).cpu() #一次元に変換

        #3.得られたポリシーによって、ノードからのアクションを初期化。(prepare_init_actions())
        self.prepare_init_actions(self.root_node, root_prediction_p)

        #4.kr_update()を行う。(これはnode.cppの関数であり、とりあえず省略)
        self.root_node.kr_update(root_leaf_black_eval)

        #5.定数のプレイアウト回数分mctsを用いたsimulationをする。(play_simulation())
        n = 0
        for _ in range(num_playout):
            #rint("aaaa: ", self.root_node.m_v)
            #rint("child: ", self.root_node.m_children)
            #rint("len: ", len(self.root_node.m_children))
            cur_env = deepcopy(self.root_env)
            self.play_simulation(0, cur_env, self.root_node)
            remaining_time = end_time - time.time()
            #print("remaining_time: ", remaining_time)
            if remaining_time <= 0:
                break
            n += 1
        
        print("探索回数：", n)
        print("root_node: ", self.root_node.m_visits)

        thinking_time = time.time() - st

        #6得られた最善手を出力する。(sample_best_action())
        self.print_winrate()
        return self.old_sample_best_action()




    def predict(self, cur_env):
        #rint(cur_env.game_state["num_shot"])
        if self.root_env.game_state["end"] != cur_env.game_state["end"]:
            #rint("predict: ", cur_env.game_state["score"][self.root_env.game_state["end"]])
            leaf_dist_v = [0] * N_SCORE
            leaf_dist_v[score_to_idx(cur_env.game_state["score"][self.root_env.game_state["end"]])] = 1
            #rint(cur_env.game_state["score"][self.root_env.game_state["end"]])
            leaf_black_eval = dist_v_to_exp_v(leaf_dist_v)
            #rint("pppppppppp: ", leaf_black_eval)
            return None, leaf_black_eval
        else:
            #if len(cur_env.game_state["stones"]) == 15:
            input_data = generate_input_planes(stones=cur_env.game_state["stones"], scores=cur_env.game_state["scores"], end=cur_env.game_state["end"], shot=cur_env.game_state["num_shot"])
            input_plane = torch.tensor(input_data.reshape(1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE)).to(self.device)
            prediction_p, prediction_v = self.network.forward_with_softmax2(input_plane)
            if cur_env.game_state["WhiteToMove"]:
                prediction_v = torch.flip(prediction_v, [1])
            prediction_v = prediction_v.cpu()
            leaf_black_eval = dist_v_to_exp_v(prediction_v.numpy())
        return prediction_p, leaf_black_eval       

    def prepare_init_actions(self, node, prediction_p):
        factor = 1.0
        prev_init_action_prob = 0.0

        #初期アクションのサンプリング
        for _ in range(self.num_init):
            init_action_id = np.random.choice(np.arange(len(prediction_p)), p=self.apply_temperature(prediction_p, self.init_temperature))
            init_action_prob = prediction_p[init_action_id]

            factor = factor * (1.0 - prev_init_action_prob)
            prev_init_action_prob = init_action_prob

            #node.add_init_info(init_action_id, init_action_prob * factor)
            node.m_init_infos.append((init_action_id, (init_action_prob * factor)))

            prediction_p[init_action_id] = 0.0
            prediction_p = prediction_p + 1.0E-6
            prediction_p = prediction_p / prediction_p.sum() 



    def apply_temperature(self, distribution, temperature):
        if temperature < 0.1:
            probabilities = np.zeros(distribution.shape[0])
            probabilities[np.argmax(distribution)] = 1.
        else:
            log_probabilities = np.log(distribution)
            log_probabilities = log_probabilities * (1 / temperature)
            # scale probabilities to a more numerically stable range (in log space)
            log_probabilities = log_probabilities - log_probabilities.max()
            # convert back from log space
            probabilities = np.exp(log_probabilities)
            # re-normalize the distribution
            probabilities = probabilities / probabilities.sum()
            probabilities = probabilities.numpy()
        return probabilities 


    def play_simulation(self, cur_depth, cur_env, cur_node):
        #入力は探索木の深さ、simulate環境、ノード
        is_expanded = False
        #rint(cur_env.game_state["num_shot"], cur_env.game_state["WhiteToMove"])
        #エンドの終了まで、または既定の深さまでシミュレーションをしたら。
        if (self.root_env.game_state["end"] != cur_env.game_state["end"]) or (cur_depth == self.max_depth):
            #rint("00000000")
            prediction_p, leaf_black_eval = self.predict(cur_env)
            #return None, False
            self.kari += 1
            #rint("aalalala: ", leaf_black_eval)
            cur_node.kr_update(leaf_black_eval)
            return leaf_black_eval, True

        num_child = len(cur_node.m_children)
        num_total_child_visits = cur_node.m_visits
        if num_child < self.num_init:
            init_action_id, init_action_prob = cur_node.m_init_infos[num_child]
            init_action = index_to_shot(init_action_id)
            expanded_node = cur_node.add_node(init_action[0], init_action[1], init_action[2], init_action_prob)
            self.num_node += 1
            next_env = self.get_env_change(cur_env, expanded_node)
            prediction_p, leaf_black_eval = self.predict(next_env) 
            if prediction_p is not None: #エンドが終了してない場合
                prediction_p = prediction_p.view(-1)
                self.prepare_init_actions(expanded_node, prediction_p)
            expanded_node.kr_update(leaf_black_eval)
            is_expanded = True
        else:
            selected_node = cur_node.ucb_select(cur_env.game_state["WhiteToMove"], self.ucb_const)
            #rint(self.pw_const * (num_child ** 0.8) - num_total_child_visits,"num_total_child_visits: ", num_total_child_visits, "num_child: ", num_child)
            #rint("num_total_child_visits: ", num_total_child_visits, "num_child: ", num_child, "diff: ", self.pw_const * (num_child ** 0.8) - num_total_child_visits)
            if num_total_child_visits < self.pw_const * (num_child ** 2): # Progressive Widening
                #rint("num_total_child_visits: ", num_total_child_visits, "num_child: ", num_child, "diff: ", self.pw_const * (num_child ** 2) - num_total_child_visits)
                #rint("num_total_child_visits: ", num_total_child_visits, "num_child: ", num_child)
                # select
                """
                if(cur_env.game_state["num_shot"] == 15):
                    selected_env = self.get_env_15(selected_node)
                else:
                    selected_env = self.get_env(selected_node)
                    """
                selected_env = self.get_env_change(cur_env, selected_node)
                leaf_black_eval, is_expanded = self.play_simulation(cur_depth + 1, selected_env,  selected_node)
            if not is_expanded:
                # expand with continuous action sample
                sample_move = selected_node.sample_move(self.num_sample, self.l)
                expanded_node = cur_node.add_node(sample_move[0], sample_move[1], sample_move[2], 0)
                self.num_node += 1

                next_env = self.get_env_change(cur_env, expanded_node)
                prediction_p, leaf_black_eval = self.predict(next_env)
                if prediction_p is not None: #エンドが終了してない場合
                    prediction_p = prediction_p.view(-1)
                    self.prepare_init_actions(expanded_node, prediction_p)
                expanded_node.kr_update(leaf_black_eval)
                is_expanded = True
        cur_node.kr_update(leaf_black_eval)
        return leaf_black_eval, is_expanded


    def get_env(self, cur_node):
        cur_node_key = self.node_key(cur_node)
        if cur_node_key not in self.env_dict.keys():
            prev_env = self.env_dict[self.node_key(cur_node.m_parent)]
            cur_env = deepcopy(prev_env)
            #cur_env.step_without_rand(*cur_node.action)
            cur_env = cur_env.step(cur_node.m_move[0], cur_node.m_move[1], cur_node.m_move[2])
            self.env_dict[cur_node_key] = cur_env
        return self.env_dict[cur_node_key]
    
    def get_env_15(self, cur_node):
        prev_env = self.env_dict[self.node_key(cur_node.m_parent)]
        cur_env = deepcopy(prev_env)
        cur_env = cur_env.step(cur_node.m_move[0], cur_node.m_move[1], cur_node.m_move[2])
        return cur_env

    def get_env_change(self, cur_env, cur_node):
        if(cur_env.game_state["num_shot"] == 15):
            return self.get_env_15(cur_node)
        else:
            print("else")
            return self.get_env(cur_node)


    def node_key(self, cur_node):
        key = ""
        while cur_node.m_parent is not None:
            key = f"({cur_node.m_move[0]:.4f} {cur_node.m_move[1]:.4f} {cur_node.m_move[2]})->{key}"
            cur_node = cur_node.m_parent
        return key


    def old_sample_best_action(self):
        kns = np.array([c.m_visits for c in self.root_node.m_children])
        kns = kns / sum(kns)
        
        best_id = np.random.choice(np.arange(len(kns)), p=self.apply_temperature(kns, self.out_temperature))
        print("v: ", self.root_node.m_children[best_id].m_v, "visit: ", self.root_node.m_children[best_id].m_visits)
        return self.root_node.m_children[best_id].m_move

    def print_winrate(self):
        min = -10
        num = 0
        sama = 0
        for child in self.root_node.m_children:
            sama += child.m_visits
            if(min < (child.m_v / child.m_visits)):
                min = (child.m_v / child.m_visits)
                print((child.m_v / child.m_visits), "aaa: ", child.m_visits)
                #rint(child.get_eval(True))
                minchild = child
                minnum = num
            num += 1
        print(min)
        print(minchild)
        print("sum: ", sama)
        print("amari: ", self.kari)
        print(self.root_node.m_children[minnum])
        print("winrate: ", self.root_node.m_children[minnum].m_move)

    def sample_best_action(self):
        min = -10
        num = 0
        for child in self.root_node.m_children:
            if (min < (child.m_v / child.m_visits)):
                min = child.m_v / child.m_visits
                minchild = child
                minnum = num
            num += 1
        print("ucb: ", self.root_node.ucb_select2(self.root_env.game_state["WhiteToMove"], self.ucb_const).m_move)
        return self.root_node.m_children[minnum].m_move