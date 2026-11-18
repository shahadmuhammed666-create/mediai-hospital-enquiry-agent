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
3. Create a Gemini API key in
   [Google AI Studio](https://aistudio.google.com/api-keys), then enter it in
   Render as `GEMINI_API_KEY`. Keep it in Render's environment settings; never
   put it in the repository or share it in chat.
4. Deploy the service. When it finishes, Render will provide a public URL you
   can share.

The web app uses `gemini-3.8-flash` by default. Set `GEMINI_MODEL` in Render to
use another model available to your Google AI Studio project. Gemini's free
tier has usage limits, and Google may use free-tier prompts to improve its
products. Do not send identifying or sensitive patient information.

The free hosting plan may sleep when idle and has resource limits. Every visitor
uses the same Gemini API key and quota. Chat messages are held only in the
visitor's open page and are sent to Google to produce answers. The backend does
not persist them. Free-tier quotas can be exhausted; the app may stop responding
until the quota resets or another eligible model is selected.
