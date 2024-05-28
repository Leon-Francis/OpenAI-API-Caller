# OpenAI-API-Caller

Self used OpenAI API caller, supporting parallel; interrupt and continue; parsing with regex.

## How to use

### Step 1
```
# The Python version needs to be 3.10 or higher.
pip install -r requirement.txt
```
### Step 2

Create a new file called .env and set your proxy (if needed) and api key in it.

```
# Example .env file
http_proxy = "..."
https_proxy = "..."
API_KEY = "..."
```

### Step 3

```
from openai_api_caller import openai_api_caller
llm_response = openai_api_caller(prompts, model_name)
```

See example.py for more detials.