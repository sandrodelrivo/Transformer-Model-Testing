from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
import openai
import asyncio
import jsonlines
import json
import asyncio
import time

openai.api_key = ""


class GPTAgent(Agent):

    num_correct = 0
    num_questions = 0

    is_First = True
    id = ""
    prompt = ""

    model = ""

    #prompt_seed =  "Sandra went back to the garden.\nSandra went to the hallway.\nWhere is Sandra?\nhallway\n\n"

    prompt_seed = ''

    # initialize by setting id
    def __init__(self, opt, mode):
        self.id = 'RepeatLabel'
        self.mode = mode

    # store observation for later, return it unmodified
    def observe(self, observation):
        self.observation = observation

        if self.is_First:
            self.id = observation["id"]
            self.is_First = False
            self.prompt = self.prompt_seed

        return observation

    # return label from before if available
    def act(self):

        print(self.observation)

        self.num_questions += 1

        reply = {'id': self.id}

        text = self.observation['text']

        if self.observation['episode_done']:
            self.prompt = self.prompt_seed

        self.prompt += text + " "

        # response = self.get_generic_response(self.observation['text'])
        # response = self.get_answer_response(self.observation['text'])
        time.sleep(1.1)
        response = self.get_oneshot_response(self.prompt)
        answer = self.observation["labels"][0]

        if (answer in response):
            self.num_correct += 1

        self.prompt += answer.strip() + "\n\n"

        reply['text'] = "".join(response)

        return reply


    def get_generic_response(self, text):

        prompt = text + "\n"


        response = openai.Completion.create(model=self.model, prompt=prompt, max_tokens=15, stop="\n")


        return response.choices[0].text.lower()

    def get_oneshot_response(self, prompt):

        print("PROMPT: ", prompt)

        response = openai.Completion.create(model=self.model, prompt=prompt, max_tokens=5, stop="\n")

        answer = response.choices[0].text.lower()
        answer = answer.replace(".", "")
        answer = answer.replace("?", "")

        print("ANSWER:", answer)

        return answer

    @staticmethod
    def get_answer_response(text):

        split_text = text.split("\n")

        question = split_text.pop()

        info = "".join(split_text)

        print("INFO:", info)
        print("QUESTION:", question)

        doc = {
            "text": info,
            "metadata": "information about the person's location"
        }

        with open("data.jsonl", 'w') as f:
            f.write(json.dumps(doc))

        response = openai.File.create(file=open("data.json"), purpose='answers')

        answer = openai.Answer.create(
            search_model="ada:ft-uchicago-2021-12-05-19-19-23",
            model="ada:ft-uchicago-2021-12-05-19-19-23",
            question="file-2ksWL61f0Q5c5vCYOLwUuhPk",
            file=response.id,
            examples_context="In 2017, U.S. life expectancy was 78.6 years.",
            examples=[["What is human life expectancy in the United States?", "78 years."]],
            max_rerank=10,
            max_tokens=10,
            stop=["\n", "<|endoftext|>"]
        )

        print("ANSWER:", answer)
        return answer


def main():
    parser = ParlaiParser()
    opt = parser.parse_args()

    print(opt)

    agent = GPTAgent(opt, 'default')
    world = create_task(opt, agent)

    for _ in range(100):
        world.parley()
        print(world.display())
        if world.epoch_done():
            print('EPOCH DONE')
            break

    print("NUM QUESTIONS:", agent.num_questions, "NUM CORRECT:", agent.num_correct,
          "PERCENT ACCURACY:", agent.num_correct/agent.num_questions)


main()
