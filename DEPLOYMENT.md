# MediAi public demo

MediAi is a public demonstration, not a medical service. It uses sample doctor
and availability data from this project. Do not enter identifying or sensitive
patient information. The website does not enable appointment booking or write
chat messages to the project's logs or memory file.

## Deploy on Render

1. Create a GitHub repository for this project and push the project files.
   Do not upload `.venv`, logs, memory files, or API keys. The `.gitignore`
   file excludes those local files.
2. In Render, choose **New** → **Blueprint** and connect the GitHub repository.
   Render will read `render.yaml` and configure the Python web service.
3. Create an OpenAI API key in the
   [OpenAI API dashboard](https://platform.openai.com/api-keys), then enter it
   in Render as `OPENAI_API_KEY`. Keep it in Render's environment settings;
   never put it in the repository or share it in chat.
4. Deploy the service. When it finishes, Render will provide a public URL you
   can share.

The web app uses `gpt-4o-mini` by default. Set `OPENAI_MODEL` in Render to use
another model available to your OpenAI API project. OpenAI API usage and billing
are separate from a ChatGPT subscription; configure API billing and usage limits
in your OpenAI account.

The free hosting plan may sleep when idle and has resource limits. Every visitor
uses the same OpenAI API key and quota. Chat messages are held only in the
visitor's open page and are sent to OpenAI to produce answers. The backend does
not persist them.
