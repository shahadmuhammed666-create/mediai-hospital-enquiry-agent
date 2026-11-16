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
3. When prompted, enter your Gemini API key for `GEMINI_API_KEY`. Keep it in
   Render's environment settings; never put it in the repository.
4. Deploy the service. When it finishes, Render will provide a public URL you
   can share.

The site uses `gemini-3.6-flash` by default. Change the `GEMINI_MODEL`
environment variable in Render if you need another model ID available to your
Gemini API project.

The free hosting plan may sleep when idle and has resource limits. Every visitor
uses the same Gemini API key and quota, so the Gemini free-tier request limit
may be reached quickly. Chat messages are held only in the visitor's open page
and are sent to Gemini to produce answers. The backend does not persist them.
