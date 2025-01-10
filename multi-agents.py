import os,re

# Disable the OpenTelemetry SDK that crewAI uses
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew

import importlib.metadata

def debug(*msg):
    print('\033[93m',msg,'\033[0m')
          
debug('Using crewai version', importlib.metadata.version("crewai"))

# 1. Image Classifier Agent (to check if the image is an animal)

classifier_agent = Agent(
    role="Image Classifier Agent",
    goal="Determine if the image is of an animal or not",
    backstory="""
        You have an eye for animals! Your job is to identify whether the input image is of an animal
        or something else.
    """,
    llm='ollama/llava:7b'  # Model for image-related tasks
)

# 2. Animal Description Agent (to describe the animal in the image)
description_agent = Agent(
    role="Animal Description Agent {image_path}",
    goal="Describe the animal in the image",
    backstory="""
        You love nature and animals. Your task is to describe any animal based on an image.
    """,
    llm='ollama/llava:7b'  # Model for image-related tasks
)

# 3. Information Retrieval Agent (to fetch additional info about the animal)
info_agent = Agent(
    role="Information Agent",
    goal="Give compelling information about a certain animal",
    backstory="""
        You are very good at telling interesting facts.
        You don't give any wrong information if you don't know it.
    """,
    llm='ollama/llama3.2'  # Model for general knowledge retrieval
)

# Task 1: Check if the image is an animal
task1 = Task(
    description="Classify the image ({image_path}) and tell me if it's an animal.",
    expected_output="If it's an animal, say 'animal'; otherwise, say 'NOT an animal'. Do not say anything else.",
    agent=classifier_agent
)

# Task 2: If it's an animal, describe it
task2 = Task(
    description="Describe the animal in the image.({image_path})",
    expected_output="Give a brief description of the animal.",
    agent=description_agent
)

# Task 3: Provide more information about the animal
task3 = Task(
    description="Give additional information about the described animal.",
    expected_output="Provide at least 3 interesting facts or information about the animal, as a numbered list.",
    agent=info_agent
)

# Crews to manage the agents and tasks
classify = Crew(
    agents=[classifier_agent],
    tasks=[task1],
    verbose=False
)

describeAndInfo = Crew(
    agents=[description_agent,info_agent],
    tasks=[task2, task3],
    verbose=True
)

def runTasks(filename):
    inputs = {'image_path': image_path}
    debug('checking for animals',image_path)
    output1 = classify.kickoff(inputs=inputs)
    if re.search(r'NOT an animal', str(output1)):
        debug(image_path,'is not an animal, stopping here')
    else:
        describeAndInfo.kickoff(inputs=inputs)

# Execute the tasks with the provided image path
image_folder = './images'
for filename in sorted(os.listdir(image_folder)):
    image_path=os.path.join(image_folder, filename)
    debug('processing',image_path)
    runTasks(image_path)