
#### SELF PLAY
EPISODES = 1 # 30  # 나중에 늘리기 
MCTS_SIMS = 4096 # 50
MEMORY_SIZE = 20000 # 30000
TURNS_UNTIL_TAU0 = 5 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.1
ALPHA = 0.9
THREADS = 32


BOARD_SIZE = 9


#### RETRAINING
BATCH_SIZE = 256 # 256
EPOCHS = 20
REG_CONST = 0.0001
LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_LOOPS = 10
'''
HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]
'''

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	]

#### EVALUATION
EVAL_EPISODES = 2 #20
SCORING_THRESHOLD = 1.3
