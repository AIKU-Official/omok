import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import torch

import matplotlib.pyplot as plt
# from IPython import display
import pylab as pl

from config import BATCH_SIZE

import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class User():
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        #action = input('Enter your chosen action: ')

        # if input is e5 -> 4,4
        # min = 0,0 (0), max = 8,8 (80)
        action_str = str(input('Enter your action: '))
        action = (int(action_str[1:]) - 1) * config.BOARD_SIZE + (ord(action_str[0]) - 97)
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return (action, pi, value, NN_value)


class Agent():
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model, device):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model
        self.device = device

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self):
        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(lg.logger_mcts)
        lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
        leaf.state.render(lg.logger_mcts)

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def act(self, state, tau):
        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation
        '''for sim in range(self.MCTSsimulations):
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('***************************')
            self.simulate()'''
        self.parallel_simulate()

        #### get action values
        pi, values = self.getAV(1)

        #### pick the action
        action, value = self.chooseAction(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = -self.get_preds(nextState)[0]

        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

        return (action, pi, value, NN_value)

    def get_preds(self, state):
        inputToModel = self.model.convertToModelInput(state)
        inputToModel = inputToModel.to(self.device)  # GPU로 이동 (필요한 경우)
        with torch.no_grad():
            value, probs = self.model.predict(inputToModel)
        value = value.item()  # 텐서에서 스칼라 값 추출
        probs = probs.detach().cpu().numpy()[0]  # 텐서를 NumPy 배열로 변환
        allowedActions = state.allowedActions
        mask = np.zeros(self.action_size)
        mask[allowedActions] = 1
        probs = probs * mask  # Masking invalid actions
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            # 모든 확률이 0인 경우, 균등 분포 사용
            probs = mask / np.sum(mask)
        return value, probs, allowedActions

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        lg.logger_mcts.info('------EVALUATING LEAF------')

        if done == 0:
            value, probs, allowedActions = self.get_preds(leaf.state)
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction(action)
                if newState.id not in self.mcts.tree:
                    node = mc.Node(newState)
                    self.mcts.addNode(node)
                    lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
                else:
                    node = self.mcts.tree[newState.id]
                    lg.logger_mcts.info('existing node...%s...', node.id)

                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))
        else:
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

        return ((value, breadcrumbs))

    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory):
        lg.logger_mcts.info('******RETRAINING MODEL******')

        for i in range(5):
            ltmemory[i]['state'].printBoard()
            print(ltmemory[i]['AV'], ltmemory[i]['value'])

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = torch.tensor(np.array([
                self.model.convertToModelInput(row['state']).numpy() 
                for row in minibatch
            ]), dtype=torch.float32, device=self.device)

            training_states = training_states.squeeze(1)
            
            # 타겟 값들을 하나의 텐서로 결합
            values = torch.tensor([row['value'] for row in minibatch], dtype=torch.float32, device=self.device)
            policies = torch.tensor([row['AV'] for row in minibatch], dtype=torch.float32, device=self.device)

            print(values)
            
            # values를 (batch_size, 1) 형태로, policies를 (batch_size, action_size) 형태로 만듦
            values = values.view(-1, 1)
            training_targets = torch.cat([values, policies], dim=1)

            print(training_states.shape, training_targets.shape)

            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=BATCH_SIZE)
            #lg.logger_mcts.info('NEW LOSS %s', fit.history)

            #self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            #self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            #self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

        '''
        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')

        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

        # display.clear_output(wait=True)
        # display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(1.0)
        '''

        print('\n')
        # self.model.printWeightAverages()

    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds

    def buildMCTS(self, state):
        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def changeRootMCTS(self, state):
        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]

    def parallel_simulate(self, num_processes=config.THREADS):
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 각 프로세스가 동일한 시뮬레이션 횟수를 수행하도록 분배
            sims_per_process = self.MCTSsimulations // num_processes
            remaining_sims = self.MCTSsimulations % num_processes
            
            # 프로세스별 시뮬레이션 횟수 설정
            process_sims = [sims_per_process + (1 if i < remaining_sims else 0) 
                        for i in range(num_processes)]
            
            # 병렬 시뮬레이션 실행
            futures = [executor.submit(self._process_simulation, sims) 
                    for sims in process_sims]
            
            # 결과 수집 및 통계 합산
            for future in futures:
                sim_tree = future.result()
                self.mcts.merge_with(sim_tree)

    def _process_simulation(self, num_sims):
        """각 프로세스에서 실행되는 시뮬레이션"""
        print('!', end='')
        # root 노드의 복사본 생성
        root_copy = mc.Node(self.mcts.root.state)
        mcts_copy = mc.MCTS(root_copy, self.cpuct)
        
        # root 노드의 엣지 정보 복사
        for action, edge in self.mcts.root.edges:
            new_edge = mc.Edge(root_copy, edge.outNode, edge.stats['P'], action)
            root_copy.edges.append((action, new_edge))
        
        for _ in range(num_sims):
            leaf, value, done, breadcrumbs = mcts_copy.moveToLeaf()
            value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)
            mcts_copy.backFill(leaf, value, breadcrumbs)
        
        return mcts_copy.tree