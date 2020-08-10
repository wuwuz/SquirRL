import mdptoolbox
from environment import SM_env_with_stale
from environment import SM_env

env = SM_env_with_stale(max_hidden_block = 5, attacker_fraction = .15, follower_fraction = 0)
aux_env = SM_env(max_hidden_block = 5, attacker_fraction = .15, follower_fraction=0)
P, R = env.get_MDP_matrix()
solver = mdptoolbox.mdp.PolicyIteration(P, R, 0.99)
solver.run()
optimal_policy = solver.policy
print(optimal_policy[aux_env._vector_to_index((4,0,"normal"))])