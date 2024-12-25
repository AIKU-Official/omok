# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload

# PyTorch 관련 모듈 임포트
import torch

# 필요한 모듈 임포트
from game import Game, GameState
from agent import Agent
from memory import Memory
from model1 import Residual_CNN
from funcs1 import playMatches, playMatchesBetweenVersions
import torch.multiprocessing as mp

from tqdm import tqdm

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle

import csv
import pandas as pd

def run():
    # 'device' 변수 정의
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(device)

    lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
    lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
    lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

    # 환경 설정
    env = Game()

    # 기존 신경망을 로드하는 경우, config 파일을 루트로 복사
    if initialise.INITIAL_RUN_NUMBER is not None:
        copyfile(run_archive_folder  + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py', './config.py')

    import config

    ######## 메모리 로드 ########

    if initialise.INITIAL_MEMORY_VERSION is None:
        memory = Memory(config.MEMORY_SIZE)
    else:
        print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
        memory = pickle.load(open(
            run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) +
            "/memory/memory" + str(initialise.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))

    ######## 모델 로드 ########

    #print("action_size", env.action_size)
    # config 파일로부터 훈련되지 않은 신경망 객체 생성
    current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                            (2,) + env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS)
    best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                        (2,) + env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS)

    # 기존 신경망을 로드하는 경우, 해당 모델의 가중치를 설정
    if initialise.INITIAL_MODEL_VERSION is not None:
        best_player_version = initialise.INITIAL_MODEL_VERSION
        print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
        best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
        current_NN.load_state_dict(best_NN.state_dict())
    else:
        best_player_version = 0
        best_NN.load_state_dict(current_NN.state_dict())

    # 모델을 device로 이동
    current_NN.to(device)
    best_NN.to(device)

    # config 파일을 실행 폴더로 복사
    copyfile('./config.py', run_folder + 'config.py')

    print('\n')

    ######## 플레이어 생성 ########

    mp.set_start_method('spawn', force=True)

    current_player = Agent('current_player', env.state_size, env.action_size,
                        config.MCTS_SIMS, config.CPUCT, current_NN, device)
    best_player = Agent('best_player', env.state_size, env.action_size,
                        config.MCTS_SIMS, config.CPUCT, best_NN, device)
    # user_player = User('player1', env.state_size, env.action_size)
    iteration = 0

    idx = 0

    data = pd.read_csv('training_data_9by9.csv')

    while True:
        iteration += 1
        reload(lg)
        reload(config)

        print('ITERATION NUMBER ' + str(iteration))

        lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
        print('BEST PLAYER VERSION ' + str(best_player_version))

        ######## 자기 플레이 (Self-Play) ########
        '''print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
        _, memory, _, _ = playMatches(
            best_player, best_player, config.EPISODES, lg.logger_main,
            turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)
        print('\n')'''

        print('Converting data to memory...')

        for _ in tqdm(range(1000)):
            # load idx+ith row
            row = data.iloc[idx]
            row = row.to_dict()
            winner = 1 if row['winner'] == 'black' else -1
            player = 1
            moves = row['moves']
            moves = moves.split(' ')
            board = np.zeros(config.BOARD_SIZE**2, dtype=int)
            for move in moves:
                pos = ord(move[0])-ord('d') + (int(move[1:])-4)*config.BOARD_SIZE
                av = np.zeros(config.BOARD_SIZE**2, dtype=int)
                av[pos] = 1
                
                state = GameState(board.copy(), player)
                memory.stmemory.append({
				'board': state.board
				, 'state': state
				, 'id': state.id
				, 'AV': av.copy()
				, 'playerTurn': state.playerTurn
                , 'value': winner
				})

                board[pos] = player

                player = -player
                winner = -winner

            idx += 1

        memory.commit_ltmemory()
        memory.clear_stmemory()

        # if len(memory.ltmemory) >= config.MEMORY_SIZE:
        if True:
            ######## 재학습 ########
            print('RETRAINING...')
            current_player.replay(memory.ltmemory)
            print('')

            if iteration % 5 == 0:
                pickle.dump(memory, open(
                    run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb"))

            lg.logger_memory.info('====================')
            lg.logger_memory.info('NEW MEMORIES')
            lg.logger_memory.info('====================')

            memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))

            for s in memory_samp:
                current_value, current_probs, _ = current_player.get_preds(s['state'])
                best_value, best_probs, _ = best_player.get_preds(s['state'])

                lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                lg.logger_memory.info('THE MCTS ACTION VALUES: %s',
                                    ['%.2f' % elem for elem in s['AV']])
                lg.logger_memory.info('CUR PRED ACTION VALUES: %s',
                                    ['%.2f' % elem for elem in current_probs])
                lg.logger_memory.info('BES PRED ACTION VALUES: %s',
                                    ['%.2f' % elem for elem in best_probs])
                lg.logger_memory.info('ID: %s', s['state'].id)
                lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))

                s['state'].render(lg.logger_memory)

            ######## 토너먼트 ########
            '''if iteration % 5 == 0:
                print('TOURNAMENT...')
                scores, _, points, sp_scores = playMatches(
                    current_player, current_player, 1, lg.logger_tourney,
                    turns_until_tau0=0, memory=None)
                print('\nSCORES')
                print(scores)
                print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
                print(sp_scores)'''



            '''
            print('\n\n')

            if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
                best_player_version += 1
                best_NN.load_state_dict(current_NN.state_dict())
                best_NN.write(env.name+'best', best_player_version)'''

        else:
            print('MEMORY SIZE: ' + str(len(memory.ltmemory)))

        ######## PyTorch 모델 저장 ########
        current_NN.write(env.name+'curr', iteration)




if __name__ == '__main__':
    mp.freeze_support()
    run()