import matplotlib.pyplot as plt
from Game import TicTacToe
from QLearning import Qlearning

game = TicTacToe(True)  # Game instance, True means training
player1 = Qlearning()   # Player1 learning agent
player2 = Qlearning()   # Player2 learning agent
game.startTraining(player1, player2)  # Start training

game.train(200000)  # Train for 200,000 iterations
game.saveStates()   # Save Q-table
game.plot_results() # Plot the results
