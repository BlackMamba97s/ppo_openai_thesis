import numpy as np
from agent.train_agent import TrainingAgent
from config import args
from environments.environments import TestEnvironment
from agent.test_agent import TestAgent

if __name__ == "__main__":
    # Initialize the agent and load trained parameters if in test mode or just start a training otherwise

    if args.mode == 'test':
        print("Starting Test mode")
        agent = TestAgent()
        agent.load_param()

        # Initialize the environment
        env = TestEnvironment()

        # Run the testing loop
        for i_ep in range(10):
            score = 0
            state = env.reset()

            for t in range(1000):
                action = agent.select_action(state)
                state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

                if args.render:
                    env.render()

                score += reward
                state = state_

                if done or die:
                    break

            print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))

    else:
        print("Starting Train mode")
        agent = TrainingAgent()
        env = TestEnvironment()


        training_records = []
        running_score = 0
        state = env.reset()
        for i_ep in range(100000):
            score = 0
            state = env.reset()

            for t in range(1000):
                action, a_logp = agent.select_action(state)
                state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                if args.render:
                    env.render()
                if agent.store((state, action, a_logp, reward, state_)):
                    print('updating')
                    agent.update()
                score += reward
                state = state_
                if done or die:
                    break
            running_score = running_score * 0.99 + score * 0.01

            if i_ep % args.log_interval == 0:
                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
                agent.save_param()
            if running_score > env.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break


