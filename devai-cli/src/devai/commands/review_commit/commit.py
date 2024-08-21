# Review code changes based on commits.

import click
from devai.util.file_processor import (
    format_files_as_string,
    list_changes,
    list_commit_messages,
    list_files,
)
from google.cloud.aiplatform import telemetry
from vertexai.generative_models import GenerativeModel

USER_AGENT = "cloud-solutions/genai-for-developers-v1.0"
model_name="gemini-1.5-pro"

source = '''
GIT DIFFS:
{}

GIT COMMITS:
{}

FINAL CODE:
{}
'''
report_qry = '''
INSTRUCTIONS:
You are senior software engineer doing a code review. You are given following information:
GIT DIFFS - new code changes
GIT COMMITS - developer written comments for new code changes
FINAL CODE - final version of the source code

GIT DIFFS show lines added and removed with + and - indicators.
Here's an example:
This line shows that code was changed/removed from the FINAL CODE section:
-            return f"file: source: [Binary File - Not ASCII Text]"
This line shows that code was changed/added in the FINAL CODE section:
+            # return f"file: source: [Binary File - Not ASCII Text]

GIT COMMITS show the commit messages provided by developer that you can use for extra context.

Using this pattern, analyze provided GIT DIFFS, GIT COMMITS and FINAL CODE section 
and write explanation for internal company change management about what has changed in several sentences with bullet points.
Use professional tone for explanation.
Only write explanation for new code changes and not for existing code in the FINAL CODE section.
Set the title to "## Code Changes Summary by Gemini ✨"
'''

@click.command(name="commit")
@click.option('-h', "--hash", required=True, type=str, default="")
@click.option('-b', "--base", required=False, type=str)
def commit(hash, base):
    """
    This function performs a review on what has changed in commits based on SHA.
    """
    base = base or hash
    refer_commit_parent = False if base != hash else True
    files = list_files(base, hash, refer_commit_parent)
    changes = list_changes(base, hash, refer_commit_parent)
    commit_messages = list_commit_messages(base, hash, refer_commit_parent)
    source_code = source.format(changes, commit_messages, format_files_as_string(files))
    code_chat_model = GenerativeModel(model_name)
    with telemetry.tool_context_manager(USER_AGENT):
        code_chat = code_chat_model.start_chat(response_validation=False)
        code_chat.send_message(report_qry)
        response = code_chat.send_message(source_code)

    click.echo(f"{response.text}")
