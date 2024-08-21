# Review code changes based on commits.

import logging

import click
import requests
from google.cloud.aiplatform import telemetry
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory

from devai.util.file_processor import (
    format_files_as_string,
    list_changes,
    list_commit_messages,
    list_files,
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


USER_AGENT = "cloud-solutions/genai-for-developers-v1.0"
model_name = "gemini-1.5-pro"

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
You are a truthful senior software engineer doing a code review. You are given following information:
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

If there is any package update in these files, do:
1. Get the correct TOOL:
   * File: `package.json`, tool: `get_npm_release_notes`
2. Invoke the TOOL:
   <tool_code>
   print(<tool_name>(package_name='<package_name>', version='<version>'))
   </tool_code>
3. Summarize the changes in the release notes. Do not make up the changes.

Set the title to "## Code Changes Summary by Gemini ✨"
'''


class GetReleaseNotesInput(BaseModel):
    package_name: str = Field(description="The name of the package.")
    version: str = Field(description="The version of the package.")


@tool(args_schema=GetReleaseNotesInput)
def get_js_release_notes(
    package_name: str,
    version: str,
) -> str:
    """
    Get list of changes for a version upgrade of a dependency/package listed in package.json file.
    This is done by getting the release notes for the version upgrade of the package.
    """
    try:
        resp1 = requests.get(f"https://registry.npmjs.org/{package_name}")
        npm_info = resp1.json()
        githubUrl = npm_info["repository"]["url"]
        githubUser = githubUrl.split("/")[3]
        githubRepoName = githubUrl.split("/")[4].split(".")[0].split("#")[0]

        resp2 = requests.get(
            f"https://api.github.com/repos/{githubUser}/{githubRepoName}/releases/tags/v{version}"
        )
        release_info = resp2.json()
        return f"""Release notes for {package_name} {version}:
{release_info["body"]}
"""
    except Exception as e:
        logging.error(f"An error occurred while retrieving release notes '{package_name}@{version}': {e}")
        return ""


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

    llm = ChatVertexAI(
        model_name=model_name,
        safety_settings=safety_settings,
    )
    tools = [get_js_release_notes]

    # Construct the tool calling agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", report_qry),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    with telemetry.tool_context_manager(USER_AGENT):
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        response = agent_executor.invoke({"input": source_code})

    click.echo(f"{response['output']}")
