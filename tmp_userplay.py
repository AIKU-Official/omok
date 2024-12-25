
import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload

# PyTorch 관련 모듈 임포트
import torch

# 필요한 모듈 임포트
from game import Game, GameState
from agent import Agent, User
from memory import Memory
from model1 import Residual_CNN
from funcs1 import UserplayMatches, playMatchesBetweenVersions
import torch.multiprocessing as mp
import config


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

    

    ######## 모델 로드 ########
    best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                        (2,) + env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS)
    
    model_path = "/home/aikusrv01/omok/torch_omok_test/DeepReinforcementLearning/models/gomokucurr_0025.pt"
    best_NN.load_state_dict(torch.load(model_path, map_location=torch.device(device)))  # CPU에서 모델 로드

    best_NN.to(device)

    # config 파일을 실행 폴더로 복사
    copyfile('./config.py', run_folder + 'config.py')

    print('\n')

    ######## 플레이어 생성 ########

    mp.set_start_method('spawn', force=True)

    current_player = User("User", env.state_size, env.action_size)  # 유저 플레이어 초기화
    best_player = Agent('best_player', env.state_size, env.action_size,
                        config.MCTS_SIMS, config.CPUCT, best_NN, device)

    print('TOURNAMENT...')
    scores, _, points, sp_scores = UserplayMatches(
        best_player, current_player, 1, lg.logger_tourney,
        turns_until_tau0=0, memory=None)
    print('\nSCORES')
    print(scores)
    print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
    print(sp_scores)




if __name__ == '__main__':
    mp.freeze_support()
    run()