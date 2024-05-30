from openai_api_caller import openai_api_caller
import logging

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)


import sys
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def main():
    prompts = [
        'What is your favorite color? A: Yellow, B: Blue, C: Red, D: Green\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite food? A: Pizza, B: Hamburger, C: Sushi, D: Pasta\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite movie? A: Titanic, B: Avatar, C: Inception, D: The Dark Knight\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite song? A: Bohemian Rhapsody, B: Stairway to Heaven, C: Imagine, D: Hotel California\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite book? A: To Kill a Mockingbird, B: 1984, C: The Great Gatsby, D: Pride and Prejudice\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite animal? A: Dog, B: Cat, C: Elephant, D: Lion\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite place? A: Paris, B: New York, C: Tokyo, D: London\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite hobby? A: Reading, B: Writing, C: Painting, D: Singing\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite sport? A: Soccer, B: Basketball, C: Tennis, D: Golf\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
        'What is your favorite game? A: Chess, B: Poker, C: Monopoly, D: Scrabble\nAnswer Format:\nMy answer: A/B/C/D\nReason: Explain why.',
    ]
    regex_pattern = r'My answer:\s*(A|B|C|D)\s*Reason:\s*(.*)'
    llm_response = openai_api_caller(
        prompts,
        model_name='gpt-3.5-turbo-0125',
        system_prompts=None,
        saved_path='output.csv',
        regex_pattern=regex_pattern,
        max_tokens=128,
        parallel_num=3
    )

if __name__ == '__main__':
    main()