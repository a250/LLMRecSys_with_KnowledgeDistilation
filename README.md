# LLMUserEncoder

This is a part of ReSys research aimed at calculating user's embeddings based on their interactions with items using LLM capability.

## Installation

1. Clone the repository

2. Create a virtual environment and activate it

3. Install the required dependencies:
    ```sh
    poetry install
    ```

## Usecase

### Step 1. Prepare embedding profiles
Based on a descrptions of `user` and `item` in given dataset derive an embeddings of profiles by using LLM.

Basic guideline [here](profile_generation/README.md)

### Step 2. Setting up and lunching experiments with knowledge transfering

Use `knowledge_transfer_framework` to setup and lunch experiments with different models.

Basic guideline [here](knowledge_transfer_framework/README.md)
