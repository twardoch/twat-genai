# TODO

## 1. Increase Test Coverage

- Add more unit tests for core functionalities (image processing, prompt handling, config parsing).
- Implement integration tests for the engines, especially Fal, to ensure proper interaction with external APIs.
- Introduce end-to-end tests for key workflows (e.g., generating an image with a specific prompt and model).

## 2. Enhance Error Handling

- Implement more specific exception types for different error scenarios (e.g., API errors, file handling errors, invalid input).
- Improve error messaging to provide more informative details for debugging.
- Gracefully handle API rate limits and network issues with retries and backoffs.

## 3. Improve Logging

- Consistently integrate `loguru` to record events and errors.
- Log important steps in the generation process, including input parameters, engine calls, and output results.
