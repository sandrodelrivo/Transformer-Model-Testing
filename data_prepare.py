from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
import openai
import asyncio
import jsonlines
import json

# slightly hacked together way of preparing the JSONL file I need to
# run the GPT fine tuning. Uses ParlAI to generate to get the bAbi tasks
# then dumps them all into a JSONL file.


class DataPreparation(Agent):

    # initialize by setting id
    def __init__(self, opt, mode):
        self.id = 'RepeatLabel'
        self.mode = mode

    task_list = []
    is_First = True
    prompt = ""

    # store observation for later, return it unmodified
    def observe(self, observation):
        self.observation = observation

        if self.is_First:
            self.id = observation["id"]
            self.is_First = False

        return observation

    # return label from before if available
    def act(self):

        reply = {'id': self.id}

        text = self.observation['text']

        if self.observation['episode_done']:
            self.prompt = ""

        self.prompt += text + "\n"

        answer = self.observation["labels"][0]

        task = {}
        task["prompt"] = self.prompt
        task["completion"] = answer

        self.task_list.append(task)

        #print(self.task_list)

        reply['text'] = ""

        return reply



def main():

    parser = ParlaiParser()
    opt = parser.parse_args()

    agent = DataPreparation(opt, 'default')
    world = create_task(opt, agent)

    num_examples = 500

    print("Preparing Example Dataset for Task:", opt['task'], "with", num_examples, "examples")

    for _ in range(num_examples):
        world.parley()
        print(world.display())
        if world.epoch_done():
            print('EPOCH DONE')
            break

    print(agent.task_list)

    with open('task_data.jsonl', 'w') as outfile:
        for entry in agent.task_list:
            json.dump(entry, outfile)
            outfile.write('\n')

main()
